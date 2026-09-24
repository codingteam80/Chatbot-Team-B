# DocuBot v6.4.88 Response-Shaping Policy

The production answer body follows a request-first policy:

**Default:** answer only what the user asked for.

The following source fields remain available but are hidden unless explicitly
requested: Category, Analysis, Applies to, C90/C99 technical mappings, external
standard cross-references, raw amplification tables, examples, exceptions,
notes, and see-also references.

Examples are included only when the user asks for examples. Classification is
included only when the user asks for category/classification/mandatory/
required/advisory status. Summaries contain only the requirement plus a concise
rationale. Lists contain identifiers plus short requirements.

The Source card remains the normal traceability surface for document/page
evidence; the answer body does not duplicate all source metadata.
