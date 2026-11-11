"""Pistas de texto zero-shot — PENDIENTE (Pareja B, idea 5).

Qué va aquí:
1. Frases por género ("a poster of a mystery film", ...), de 3 a 5 por género.
2. Vectorizarlas con `CodificadorCLIP.codificar_texto` (una sola vez).
3. Para cada cartel, similitud coseno contra cada frase, promediada por género: 17 números.
4. Guardarlas como archivo del contrato 1, p. ej. `clip_b32__zeroshot.npz` (ids, vectores n×17),
   para usarlas en un experimento junto a los vectores de imagen:
       vectores: [clip_b32__centro, clip_b32__zeroshot]
"""
