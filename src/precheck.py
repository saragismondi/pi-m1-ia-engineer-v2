"""Pre-chequeo determinista antes del modo paralelo (mejora 4, feedback de Henry, 18/11/2026).

Problema del modo paralelo (MODERATION_PARALLEL=on, mejora B2): el modelo genera la respuesta al mismo
tiempo que el moderador decide si el ticket es un ataque. La respuesta nunca sale si el moderador bloquea,
pero el modelo ya procesó el texto y la generación se paga igual.

Sugerencia de Henry: un pre-chequeo rápido y, solo si el riesgo es bajo, generar en paralelo.

Este pre-chequeo NO llama a ninguna IA (costo 0, microsegundos): busca señales "blandas", que no alcanzan
para bloquear (eso ya lo hacen los patrones de src/guardrails.py), pero sí para no apurarse:
  - operacion_sensible: el ticket pide una operación con verificación de nivel 1 a 3 (src/verification.py).
    Son los pedidos que un atacante más quiere conseguir.
  - ofuscacion: el texto cambia al desofuscarlo (caracteres invisibles, letras de otros alfabetos,
    números por letras, letras separadas). Un cliente normal casi nunca escribe así.
  - base64: hay un bloque base64 que se decodifica como texto.
  - vocabulario_de_ataque: palabras sobre instrucciones, reglas, roles o modos, aunque no formen
    un patrón completo de inyección.
  - habla_con_la_ia: le habla al asistente o al modelo, o le pide un formato o idioma de salida.
  - autoridad_o_promesa: invoca autoridad ("soy el gerente", "por orden", "autorizo") o exige una
    promesa o garantía por escrito.
  - formato_raro: etiquetas tipo <...>, bloques de código, llaves o varias líneas.
  - link: una URL dentro del ticket.
  - largo: más de LONG_QUERY_CHARS caracteres (la consulta más larga del dataset original mide 116).

Con CUALQUIER señal, el ticket va en serie: primero el moderador, después la generación.
Sin señales, va en paralelo. El orden de las capas y la falla segura del moderador no cambian.

Limitaciones (honestas):
  - un ataque bien escrito, sin ninguna de estas señales, igual va en paralelo. Lo que cambia es el caso
    común de ataque torpe u ofuscado, que deja de generarse antes de moderar;
  - las listas de palabras se ajustaron mirando los 20 ataques de evals/adversarial.jsonl, así que el
    resultado sobre ese mismo set es optimista. La tasa de falsas alarmas sí se mide en tickets que no
    se usaron para ajustar (evals/candidatos/candidatos.csv).
"""
import re

from src.guardrails import decoded_base64, deobfuscate, normalize
from src.verification import required_verification

LONG_QUERY_CHARS = 400

# Todas sobre el texto normalizado (minúsculas, sin tildes). Palabras sueltas: no alcanzan para bloquear.
_ATTACK_WORDS = re.compile(
    r"\b(instruccion\w*|prompt\w*|reglas?|politica\w*|restriccion\w*|filtros?|ignor\w*|desobedec\w*"
    r"|simula\w*|finj\w*|fingi\w*|pretend\w*|personaje|roleplay|jailbreak|developer|desarrollador"
    r"|modo\s+(desarrollador|dios|admin\w*|debug|prueba|test|libre|sin\s+\w+)|sistema\s*:|system|assistant"
    r"|instructions?|ignore)\b"
)
_TALKS_TO_AI = re.compile(
    r"\b(asistente|bot|chatbot|ia|inteligencia artificial|modelo de lenguaje|llm|gpt|claude"
    r"|json|xml|formato|textual\w*|en ingles|in english)\b"
)
_AUTHORITY_OR_PROMISE = re.compile(
    r"\b(soy (el|la) (gerente|jefe|jefa|duen\w*|director\w*|supervisor\w*|administrador\w*)|por orden"
    r"|autoriz\w*|promet\w*|garantiz\w*|por escrito|auditoria)\b"
)
# Sobre el texto original.
_ODD_FORMAT = re.compile(r"</?\s*\w+[^>]*>|```|[{}]|\[/?\w+\]")
_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\w+")


def risk_signals(query: str) -> list[str]:
    """Señales de riesgo del ticket original. Lista vacía = riesgo bajo, se puede generar en paralelo."""
    text = normalize(query)
    signals = []
    if required_verification(query).level:
        signals.append("operacion_sensible")
    # Los emails se sacan antes: su "@" se lee como "a" al desofuscar y daría una falsa alarma.
    without_emails = _EMAIL.sub(" ", query)
    if deobfuscate(without_emails) != normalize(without_emails):
        signals.append("ofuscacion")
    if decoded_base64(query):
        signals.append("base64")
    if _ATTACK_WORDS.search(text):
        signals.append("vocabulario_de_ataque")
    if _TALKS_TO_AI.search(text):
        signals.append("habla_con_la_ia")
    if _AUTHORITY_OR_PROMISE.search(text):
        signals.append("autoridad_o_promesa")
    if _ODD_FORMAT.search(query) or query.count("\n") >= 3:
        signals.append("formato_raro")
    if _LINK.search(query):
        signals.append("link")
    if len(query) > LONG_QUERY_CHARS:
        signals.append("largo")
    return signals
