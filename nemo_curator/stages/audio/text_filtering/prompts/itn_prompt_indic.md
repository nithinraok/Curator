# Inverse Text Normalization ({language})

Convert the spoken-form transcript below in **{language}** to standard written form. Only apply conversions supported by the active-language rules: numbers, dates, times, currencies, measurements, symbols, and related structured forms become their conventional written representations. Use the native vocabulary and grammatical forms of **{language}**, with the common written-form conventions below; the active-language examples define the pattern for this row.

Return ONLY the converted text. No explanations, labels, or extra formatting.

## Constraints

- Stay in **{language}** and preserve any code-switching or script mixing. Do NOT translate.
- Preserve all non-target wording and its order, including disfluencies, repetitions, false starts, colloquial forms, mispronunciations, and grammatical errors.
- Do NOT add punctuation unless it is implied by the input or required by an allowed written-form conversion.
- Do NOT paraphrase, add, or remove words beyond the conversions defined below.
- When a number expression functions idiomatically rather than numerically, keep it in word form.

## Conversion Rules

{language_rules}

### Common Written-Form Conventions

Apply these conventions to newly converted expressions in every supported language. Preserve already-written numbers, identifiers, punctuation, and script mixing outside the conversion; do not globally reformat the transcript.

- Use ASCII digits for newly converted numbers. Group ordinary cardinal quantities and the integer part of currency amounts with Indian commas: the last group has three digits and preceding groups have two, as in `2,024`, `1,00,000`, and `1,23,456.78`. Do not group years, ordinal numbers, date/time fields, phone numbers, postal codes, house identifiers, fractions, or structured letter-number codes.
- Use native-language ordinal suffixes, preserving gender, case, and attached grammatical endings. Do not introduce English `st`, `nd`, `rd`, or `th`. Express decades with native suffixes or native decade words, not English `s`; retain a lexicalized native decade expression when no numeric suffix is established.
- Put currency symbols immediately before the amount and `%` immediately after the number: `$52`, `$249.99`, `₹1,00,000`, `3.5%`. Do not put a space between a money/percent symbol and its number. For currencies with 100 minor units per major unit, use two decimal places for an explicitly spoken minor-unit amount, including leading zeroes; carry excess minor units into the major amount. Thus 625 rupees and 2 paise is `₹625.02`, not `₹625.2`.
- Dates: keep native month names, date labels, ordinal suffixes, and the spoken month/day order. Write the day without a leading zero and the year without grouping commas. Separate components with ordinary spaces; do not insert a comma before the year. Do not invent a missing day, month, year, or era marker.
- Clocks: use `H:MM`, including `H:00` for an explicitly spoken whole hour. Use `H:MM:SS` only when seconds are supplied. Do not pad the hour; pad minutes and seconds to two digits. Keep an explicit AM/PM marker as uppercase `AM` or `PM` with one preceding space; do not infer it. A standalone day-period word stays in the active language. Interpret a bare number pair as a clock only when time context or an explicit active-language rule licenses that reading; otherwise keep its list/count interpretation.
- Durations: when explicit duration units establish elapsed time, use `H:MM`, or `H:MM:SS` when seconds are supplied. Pad minutes and seconds, not hours; fill a missing intermediate field with zero, as in `9:00:02`. A duration may have more than 23 hours and never receives an inferred AM/PM marker. The duration-unit words and connectors belong to this allowed structured conversion; preserve surrounding wording. Do not reinterpret a context-free fractional quantity as a duration.
- Phones: write domestic and service numbers as contiguous digits, preserving leading zeroes. If a country code is explicitly supplied, write it as `+<country code> <national number>`, with one space after the code. Do not invent a country code or insert 3-3-4 hyphens, grouping commas, or other separators. Preserve any separately spoken extension context.

Additional rules:
- Convert the active language's spoken structural forms to `.`, `@`, `/`, `:`, or `-` only where their symbolic use is implied. Follow the active-language rules for the listed spoken forms.
- Render recognized acronyms and structured letter-number forms shown by the active-language examples in their conventional uppercase written form; do not expand them or convert unrelated letter-name sequences.
- Convert the ordinary spoken form of zero to `0`. In phone and time contexts, also convert alternate zero readings licensed by the active-language rules to `0`.
- Interpret postal or ZIP codes digit by digit and write the resulting code as digits. Write house numbers with digits.
- Abbreviate a professional title only when it directly modifies a person's name. Keep standalone, predicative, plural, inflected, and suffix-attached profession words unchanged.
- When an active-language example or rule explicitly covers an ambiguous expression, follow it instead of the general defaults below.

## Ambiguity Resolution

- Prefer digits for quantities, measurements, ages, dates, and counts, unless the active-language rules require a lexical human-count form.
- Keep number expressions in word form when they are idiomatic, part of a proper noun, pronominal or indefinite, or a vague quantity, unless an active-language ordinal rule explicitly gives a suffix-preserving written form.
- Convert an expression denoting one quarter to `1/4` only when it functions as a true fraction; leave it in word form in temporal or financial constructions.
- Convert an expression denoting one half to `1/2` only when it functions as a true fraction; leave idiomatic uses in word form.
- In stammers and false starts, preserve the broken number-word fragments and convert only the final clean numeric expression.

Spoken-form transcript in {language}:
{text}
