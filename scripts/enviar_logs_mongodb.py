#!/usr/bin/env python3
"""Envia logs de treino para uma colecao no MongoDB.

Uso rapido:
  python scripts/enviar_logs_mongodb.py
  python scripts/enviar_logs_mongodb.py --source logs_analysis/episodes.csv
  python scripts/enviar_logs_mongodb.py --dry-run --print-json

Variaveis de ambiente aceitas:
  MONGO_URI
  MONGO_DATABASE
  MONGO_COLLECTION
  PC
    SESSAO_TREINO_ID
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    MongoClient = None


LOG_PATTERN = re.compile(
    r"^\s*(?:(?P<pc>[^|]+?)\s*\|\s*)?Ep\s*(?P<ep>\d+)\s*\|\s*"
    r"(?:Noite\s*(?P<noite>\d+)\s*\|\s*)?"
    r"(?P<resultado>[A-Za-z_]+)\s*\|\s*Passos:\s*(?P<passos>-?\d+)\s*\|\s*"
    r"(?:(?:Tempo(?:\s+EP)?):\s*(?P<tempo_minutos>-?\d+(?:[.,]\d+)?)\s*min(?:utos)?\s*\|\s*)?"
    r"Recompensa:\s*(?P<recompensa>-?\d+(?:[.,]\d+)?)\s*\|\s*"
    r"Taxa[^:]*:\s*(?P<taxa>-?\d+(?:[.,]\d+)?)%"
    r"(?:\s*\|\s*Energia\s+fim:\s*(?P<energia_fim>-?\d+(?:[.,]\d+)?)\s*%?)?"
    r"(?:\s*\|\s*Causa:\s*(?P<causa>[A-Za-z_]+))?"
    r"(?:\s*\|\s*Ocorrido:\s*(?P<ocorrido>.+?))?\s*$",
    re.IGNORECASE,
)

OCORRIDO_PATTERN = re.compile(
    r"^\s*(?:(?P<pc>[^|]+?)\s*\|\s*)?Ep\s*(?P<ep>\d+)\s*\|\s*"
    r"OCORRIDO\s*\|\s*(?P<ocorrido>.+?)\s*$",
    re.IGNORECASE,
)

TREINO_INICIADO_PATTERN = re.compile(r"^\s*Treino iniciado\s*$", re.IGNORECASE)
RESULTADOS_QUE_AVANCAM_NOITE = {
    "VITORIA",
    "VITORIA_6AM",
    "VICTORIA",
    "WIN",
    "VICTORY",
}


def carregar_env(caminho: Path = Path(".env")) -> None:
    if not caminho.exists():
        return

    for linha in caminho.read_text(encoding="utf-8").splitlines():
        texto = linha.strip()
        if not texto or texto.startswith("#") or "=" not in texto:
            continue
        chave, valor = texto.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")
        if chave and chave not in os.environ:
            os.environ[chave] = valor


def detectar_fontes_log() -> list[Path]:
    candidatos = [
        Path("logs/treino.log.txt"),
        Path("logs/treino.log"),
        Path("logs_analysis/episodes.csv"),
    ]

    fontes = [candidato for candidato in candidatos if candidato.exists()]

    eventos = sorted(
        Path("logs").rglob("events.out.tfevents*"),
        key=lambda caminho: caminho.stat().st_mtime,
        reverse=True,
    )
    fontes.extend(eventos)

    if fontes:
        return fontes

    raise FileNotFoundError(
        "Nenhum log encontrado. Esperado: logs/treino.log.txt, logs/treino.log, "
        "logs_analysis/episodes.csv ou logs/**/events.out.tfevents*"
    )


def detectar_fonte_log() -> Path:
    fontes = detectar_fontes_log()
    return fontes[0]


def para_numero(valor: str) -> int | float | str | None:
    texto = valor.strip()
    if texto == "":
        return None

    texto_normalizado = texto.replace(",", ".")
    if re.fullmatch(r"-?\d+", texto_normalizado):
        return int(texto_normalizado)
    if re.fullmatch(r"-?\d+\.\d+", texto_normalizado):
        return float(texto_normalizado)
    return texto


def para_inteiro(valor: Any) -> int | None:
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return int(valor)
    if isinstance(valor, str):
        convertido = para_numero(valor)
        if isinstance(convertido, (int, float)):
            return int(convertido)
    return None


def para_float(valor: Any) -> float | None:
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, str):
        convertido = para_numero(valor)
        if isinstance(convertido, (int, float)):
            return float(convertido)
    return None


def normalizar_resultado(valor: Any) -> str | None:
    if valor is None:
        return None

    texto = str(valor).strip()
    if not texto:
        return None

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    return texto.upper()


def chave_sessao_para_noite(registro: dict[str, Any]) -> int | str:
    for chave in ("sessao_treino_log", "run"):
        valor = para_inteiro(registro.get(chave))
        if valor is not None:
            return valor
    return "sessao_unica"


def enriquecer_registros_episodios(registros: list[dict[str, Any]]) -> None:
    noite_por_sessao: dict[int | str, int] = {}

    for registro in registros:
        if "resultado" not in registro and "result" in registro:
            registro["resultado"] = registro.get("result")

        if "sessao_treino_log" not in registro and "run" in registro:
            run = para_inteiro(registro.get("run"))
            if run is not None:
                registro["sessao_treino_log"] = run

        if "tempo_ep_minutos" not in registro:
            for chave_tempo in (
                "tempo_minutos",
                "tempo_ep_min",
                "tempo_em_minutos",
                "duracao_minutos",
            ):
                valor_tempo = para_float(registro.get(chave_tempo))
                if valor_tempo is not None:
                    registro["tempo_ep_minutos"] = valor_tempo
                    break
        else:
            valor_tempo = para_float(registro.get("tempo_ep_minutos"))
            if valor_tempo is not None:
                registro["tempo_ep_minutos"] = valor_tempo

        if para_inteiro(registro.get("ep")) is None:
            continue

        chave_sessao = chave_sessao_para_noite(registro)
        noite_atual = noite_por_sessao.get(chave_sessao, 1)
        registro["noite"] = noite_atual

        resultado_normalizado = normalizar_resultado(registro.get("resultado"))
        if resultado_normalizado is not None:
            registro["resultado"] = resultado_normalizado

        if resultado_normalizado in RESULTADOS_QUE_AVANCAM_NOITE:
            noite_por_sessao[chave_sessao] = noite_atual + 1
        elif resultado_normalizado is None:
            noite_por_sessao[chave_sessao] = noite_atual
        else:
            # Em resultado nao-vitorioso, o jogo reinicia na noite 1.
            noite_por_sessao[chave_sessao] = 1


def ler_csv_episodios(caminho: Path) -> tuple[list[dict[str, Any]], str | None]:
    registros: list[dict[str, Any]] = []
    with caminho.open("r", encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            normalizado: dict[str, Any] = {}
            for chave, valor in linha.items():
                if chave is None:
                    continue
                normalizado[chave.strip()] = para_numero(valor or "")
            registros.append(normalizado)

    return registros, None


def ler_log_treino(caminho: Path) -> tuple[list[dict[str, Any]], str | None]:
    registros: list[dict[str, Any]] = []
    pc_detectado: str | None = None
    sessao_treino_atual = 0
    sessao_pendente = False
    indice_por_sessao_ep: dict[tuple[int, int], int] = {}

    with caminho.open("r", encoding="utf-8", errors="ignore") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if TREINO_INICIADO_PATTERN.match(linha):
                # So avanca sessao quando aparecer o primeiro episodio da nova sessao.
                sessao_pendente = True
                continue

            match_ocorrido = OCORRIDO_PATTERN.match(linha)
            if match_ocorrido:
                ep_ocorrido = int(match_ocorrido.group("ep"))
                indice_registro = indice_por_sessao_ep.get((sessao_treino_atual, ep_ocorrido))
                if indice_registro is not None:
                    registros[indice_registro]["ocorrido"] = match_ocorrido.group("ocorrido").strip()

                    pc_linha_ocorrido = (match_ocorrido.group("pc") or "").strip()
                    if pc_detectado is None and pc_linha_ocorrido:
                        pc_detectado = pc_linha_ocorrido
                    if pc_linha_ocorrido:
                        registros[indice_registro]["pc"] = pc_linha_ocorrido
                continue

            match = LOG_PATTERN.match(linha)
            if not match:
                continue

            if sessao_treino_atual == 0:
                # Compatibilidade com logs sem cabecalho "Treino iniciado".
                sessao_treino_atual = 1
                sessao_pendente = False
            elif sessao_pendente:
                sessao_treino_atual += 1
                sessao_pendente = False

            pc_linha = (match.group("pc") or "").strip()
            if pc_detectado is None and pc_linha:
                pc_detectado = pc_linha

            registro: dict[str, Any] = {
                "sessao_treino_log": sessao_treino_atual,
                "ep": int(match.group("ep")),
                "resultado": match.group("resultado").upper(),
                "passos": int(match.group("passos")),
                "recompensa": float(match.group("recompensa").replace(",", ".")),
                "taxa_vitoria": float(match.group("taxa").replace(",", ".")),
            }
            ocorrido = match.group("ocorrido")
            if ocorrido:
                registro["ocorrido"] = ocorrido.strip()

            tempo_minutos = match.group("tempo_minutos")
            if tempo_minutos is not None:
                registro["tempo_ep_minutos"] = float(tempo_minutos.replace(",", "."))

            # Desfecho instrumentado (logs novos): energia no fim + tipo (skill vs. sorte).
            energia_fim = match.group("energia_fim")
            if energia_fim is not None:
                registro["energia_fim"] = float(energia_fim.replace(",", "."))
            causa = match.group("causa")
            if causa:
                registro["causa"] = causa.lower()

            if pc_linha:
                registro["pc"] = pc_linha

            registros.append(registro)
            indice_por_sessao_ep[(sessao_treino_atual, registro["ep"])] = len(registros) - 1

    return registros, pc_detectado


def ler_eventos_tensorboard(caminho: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as erro:
        raise RuntimeError(
            "Nao foi possivel ler eventos TensorBoard: pacote tensorboard ausente."
        ) from erro

    acumulador = EventAccumulator(str(caminho), size_guidance={"scalars": 0})
    acumulador.Reload()

    registros: list[dict[str, Any]] = []
    tags = sorted(acumulador.Tags().get("scalars", []))
    for tag in tags:
        for evento in acumulador.Scalars(tag):
            registros.append(
                {
                    "tag": tag,
                    "step": int(evento.step),
                    "valor": float(evento.value),
                    "tempo_evento_utc": datetime.fromtimestamp(
                        evento.wall_time,
                        tz=timezone.utc,
                    ).isoformat(),
                }
            )

    return registros, None


def ler_registros(caminho: Path) -> tuple[list[dict[str, Any]], str | None, str]:
    nome = caminho.name.lower()
    if caminho.suffix.lower() == ".csv":
        registros, pc_detectado = ler_csv_episodios(caminho)
        return registros, pc_detectado, "csv"

    if caminho.suffix.lower() == ".log" or nome.endswith(".log.txt"):
        registros, pc_detectado = ler_log_treino(caminho)
        return registros, pc_detectado, "log"

    if "tfevents" in nome:
        registros, pc_detectado = ler_eventos_tensorboard(caminho)
        return registros, pc_detectado, "tensorboard"

    raise ValueError(
        f"Fonte de log nao suportada: {caminho}. "
        "Use .csv, .log, .log.txt ou events.out.tfevents*"
    )


def montar_documentos(
    *,
    registros: list[dict[str, Any]],
    pc: str,
    sessao_treino_id: str,
) -> list[dict[str, Any]]:
    coletado_em_utc = datetime.now(timezone.utc).isoformat()

    documentos: list[dict[str, Any]] = []
    for registro in registros:
        registro_documento = dict(registro)
        pc_registro = registro_documento.pop("pc", None)
        pc_final = (
            pc_registro.strip()
            if isinstance(pc_registro, str) and pc_registro.strip()
            else pc
        )

        documento: dict[str, Any] = {
            "pc": pc_final,
            "sessao_treino_id": sessao_treino_id,
            "coletado_em_utc": coletado_em_utc,
        }
        documento.update(registro_documento)

        documentos.append(documento)

    return documentos


def obter_ultimo_sessao_treino_log_no_banco(
    *,
    uri: str,
    database: str,
    collection: str,
) -> int:
    if MongoClient is None:
        raise RuntimeError(
            "Dependencia ausente: pymongo. Instale com 'pip install pymongo'."
        )

    cliente = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        cliente.admin.command("ping")
        ultimo = cliente[database][collection].find_one(
            {"sessao_treino_log": {"$type": "number"}},
            sort=[("sessao_treino_log", -1)],
            projection={"sessao_treino_log": 1},
        )
        if not ultimo:
            return 0

        valor = ultimo.get("sessao_treino_log")
        if isinstance(valor, bool):
            return 0
        if isinstance(valor, (int, float)):
            return int(valor)
        return 0
    finally:
        cliente.close()


def aplicar_offset_sessao_treino_log(
    registros: list[dict[str, Any]],
    offset: int,
) -> None:
    if offset <= 0:
        return

    for registro in registros:
        valor = registro.get("sessao_treino_log")
        if isinstance(valor, bool):
            continue
        if isinstance(valor, (int, float)):
            registro["sessao_treino_log"] = int(valor) + offset


def resumir_erro(erro: Exception, limite: int = 170) -> str:
    texto = str(erro).replace("\n", " ").strip()
    marcador_topologia = "Topology Description:"
    if marcador_topologia in texto:
        texto = texto.split(marcador_topologia, 1)[0].strip().rstrip(",")

    if len(texto) > limite:
        texto = texto[: limite - 3].rstrip() + "..."

    return f"{erro.__class__.__name__}: {texto}"


def gerar_sessao_treino_id() -> str:
    base = (
        os.getenv("WT_SESSION")
        or os.getenv("SESSIONNAME")
        or "treino"
    ).strip()
    instante = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{base}-{instante}-{uuid.uuid4().hex[:8]}"


def enviar_para_mongodb(
    *,
    documentos: list[dict[str, Any]],
    uri: str,
    database: str,
    collection: str,
) -> list[str]:
    if MongoClient is None:
        raise RuntimeError(
            "Dependencia ausente: pymongo. Instale com 'pip install pymongo'."
        )

    if not documentos:
        raise ValueError("Nenhum documento para enviar ao MongoDB.")

    cliente = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        cliente.admin.command("ping")
        resultado = cliente[database][collection].insert_many(documentos, ordered=True)
        return [str(item_id) for item_id in resultado.inserted_ids]
    finally:
        cliente.close()


def limpar_log_textual(caminho: Path, tipo_origem: str) -> bool:
    nome = caminho.name.lower()
    if tipo_origem != "log":
        return False
    if not (caminho.suffix.lower() == ".log" or nome.endswith(".log.txt")):
        return False

    caminho.write_text("", encoding="utf-8")
    return True


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ler logs de treino (log, csv ou tfevents) e enviar um JSON para MongoDB"
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Caminho do log de origem. Se omitido, detecta automaticamente.",
    )
    parser.add_argument("--pc", help="Identificador do PC. Prioriza este valor.")
    parser.add_argument(
        "--sessao-treino-id",
        "--sessao",
        dest="sessao_treino_id",
        help=(
            "Identificador para agrupar documentos da mesma execucao. "
            "Fallback: SESSAO_TREINO_ID no .env"
        ),
    )
    parser.add_argument("--uri", help="Mongo URI. Fallback: MONGO_URI no .env")
    parser.add_argument(
        "--database",
        default=os.getenv("MONGO_DATABASE", "no_more_jumpscares"),
        help="Database de destino no MongoDB",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_COLLECTION", "treino_logs"),
        help="Collection de destino no MongoDB",
    )
    parser.add_argument(
        "--max-registros",
        type=int,
        default=0,
        help="Limita quantidade de registros enviados (0 = sem limite).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nao envia para MongoDB. Apenas valida leitura.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Exibe preview do(s) documento(s) JSON no terminal.",
    )
    parser.add_argument(
        "--strict-db-error",
        action="store_true",
        help="Se definido, falhas de conexao/envio ao MongoDB geram erro e traceback.",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Exibe detalhes completos dos erros em vez de resumo curto.",
    )
    return parser


def main() -> None:
    carregar_env()
    parser = criar_parser()
    args = parser.parse_args()

    caminhos_candidatos = [args.source] if args.source else detectar_fontes_log()
    for candidato in caminhos_candidatos:
        if not candidato.exists():
            continue

        registros, pc_detectado, _tipo_origem = ler_registros(candidato)
        if registros:
            caminho = candidato
            break
    else:
        origem = (
            args.source.as_posix()
            if args.source is not None
            else "deteccao automatica (treino.log/treino.log.txt/csv/tfevents)"
        )
        print(
            "Nenhum registro valido encontrado na origem informada "
            f"({origem}). Nada para enviar."
        )
        return

    enriquecer_registros_episodios(registros)

    if args.max_registros and args.max_registros > 0:
        registros = registros[-args.max_registros :]

    uri = (args.uri or os.getenv("MONGO_URI") or "").strip()
    if _tipo_origem == "log":
        if uri:
            try:
                ultimo_banco = obter_ultimo_sessao_treino_log_no_banco(
                    uri=uri,
                    database=args.database,
                    collection=args.collection,
                )
                aplicar_offset_sessao_treino_log(registros, ultimo_banco)
            except Exception as erro:
                if args.verbose_errors:
                    detalhe_erro = str(erro)
                    print(
                        "Aviso: nao foi possivel basear sessao_treino_log no banco. "
                        f"Mantendo numeracao local. Motivo: {detalhe_erro}"
                    )
                else:
                    print(
                        "Aviso: nao foi possivel basear sessao_treino_log no banco. "
                        "Mantendo numeracao local."
                    )
        else:
            print(
                "Aviso: MONGO_URI ausente, sessao_treino_log sera baseada "
                "apenas no arquivo local."
            )

    pc = (args.pc or os.getenv("PC") or pc_detectado or platform.node()).strip()
    sessao_treino_id = (
        (
            args.sessao_treino_id
            or os.getenv("SESSAO_TREINO_ID")
            or os.getenv("TERMINAL_SESSION_ID")
            or ""
        ).strip()
        or gerar_sessao_treino_id()
    )

    documentos = montar_documentos(
        registros=registros,
        pc=pc,
        sessao_treino_id=sessao_treino_id,
    )

    if args.print_json:
        if len(documentos) == 1:
            preview: dict[str, Any] = {
                "total_documentos": 1,
                "documento": documentos[0],
            }
        else:
            preview = {
                "total_documentos": len(documentos),
                "primeiro_documento": documentos[0],
                "ultimo_documento": documentos[-1],
            }
        print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(
            "Dry-run concluido. "
            f"Documentos prontos: {len(documentos)} | origem: {caminho.as_posix()}"
        )
        return

    if not uri:
        raise ValueError(
            "Mongo URI ausente. Passe --uri ou configure MONGO_URI no .env"
        )

    try:
        inserted_ids = enviar_para_mongodb(
            documentos=documentos,
            uri=uri,
            database=args.database,
            collection=args.collection,
        )
    except Exception as erro:
        detalhe_erro = str(erro) if args.verbose_errors else resumir_erro(erro)
        mensagem = "Falha ao enviar para MongoDB. Nada foi enviado nesta execucao."
        if args.verbose_errors:
            mensagem = f"{mensagem} Motivo: {detalhe_erro}"
        else:
            mensagem = f"{mensagem} Motivo: {detalhe_erro}"
        if args.strict_db_error:
            raise RuntimeError(mensagem) from erro

        print(f"Aviso: {mensagem}")
        return

    log_foi_limpo = limpar_log_textual(caminho, _tipo_origem)

    primeiro_id = inserted_ids[0] if inserted_ids else "n/a"
    sufixo_limpeza = " | log textual limpo" if log_foi_limpo else ""
    print(
        "Documentos enviados para MongoDB com sucesso. "
        f"quantidade={len(inserted_ids)} | primeiro_id={primeiro_id}{sufixo_limpeza}"
    )


if __name__ == "__main__":
    main()