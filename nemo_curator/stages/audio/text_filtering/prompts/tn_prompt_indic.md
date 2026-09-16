# Text Normalization ({language})

Convert the written-form transcript below in **{language}** to spoken form. **Only** apply conversions from the active-language table (numbers, dates, times, money, units, symbols, etc.). Use **{language}** conventions; the examples below define the pattern for this row.

Return ONLY normalized text. No explanations or extra formatting.

## Constraints

- Stay in **{language}**; preserve code-switching/script mix. Do NOT translate.
- Do NOT clean up speech: keep fillers, repetitions, false starts, colloquial forms, grammar errors, and all non-target wording unchanged and in order.
- Do NOT add or remove words or punctuation except as required by conversions (for example, replacing `@` with its spoken form in an email).
- Denormalize digits/symbols **only** when they match Conversion Rules. Leave idiomatic numbers, proper nouns, vague quantities, disfluencies, and text already spoken as words unchanged. Do not swap alternative spoken forms of a digit unless denormalizing a written digit sequence.
- In stammers/false starts, denormalize only the final clean numeric token.

## Conversion Rules

{language_rules}

Additional rules:
- In URLs, emails, phones, and similar structured spans, render structural symbols (`.`, `@`, `/`, `:`, `-`) with the spoken forms shown in the active-language examples.
- Keep acronyms in their natural spoken form; do not expand unrelated abbreviations.
- For `0` in phone/time contexts, use one of the natural forms shown in the active-language examples; render zip/house numbers with the active language's natural cardinal or grouped-number reading when denormalizing.

## Ambiguity

- Denormalize digit/symbol form for quantities, measurements, ages, dates, and counts.
- Skip idioms, proper nouns, vague quantities, temporal fraction words, and idiomatic fraction words.
- Convert 1/2 or 1/4 to the corresponding active-language fraction word only for true fractions.

Written-form transcript in {language}:
{text}
