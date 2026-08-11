# Prose guide

House style for everything written in this repository that a reader will see: the manuscript (`latex/main.tex`), Extended Data, figure captions, `summary.md`, the README and the Atlas site copy. It governs construction, not content. Numbers, citations and method decisions are covered by [method_notes.md](method_notes.md) and [submission_checklist.md](submission_checklist.md).

The guide exists because drafted prose (whether written fast by hand or produced by a language model) converges on a small set of faults. They are faults of rhythm rather than of grammar, so proofreading does not catch them. Each one below has been found in this manuscript at least once. The rules are stated as budgets and procedures so that a draft can be audited rather than argued about.

Read §5 before drafting and §9 before declaring any section finished.

---

## 1. Order of work

1. **Scaffold before prose.** Write the section as a bullet frame first: one bullet per claim, in the order the argument needs them. Agree the frame before any sentence is written. Prose written before the frame is settled gets rewritten wholesale, and rewriting is where the flourishes creep in.
2. **Attach the evidence to each bullet** (a `\nepi` macro name, a table, a citation) before writing. A bullet with no evidence is either cut or becomes a limitation.
3. **Draft in plain declaratives.** One claim per sentence, subject first, no connective tissue beyond "and", "but", "because". Do not attempt cadence on the first pass.
4. **Run the §9 audit.** Fix by deletion wherever possible.
5. **Read the section as a list of first clauses.** If several consecutive sentences open the same way, or if reading the openings alone conveys the argument, the middles are padding.

A section is finished when nothing can be deleted without losing a fact.

## 2. Sentence construction

- **One idea per sentence.** A second idea gets a second sentence, not a semicolon or a subordinate clause.
- **Length band 15 to 35 words.** Below 10 words is reserved for definitions and statements of scope. Above 45 words the sentence carries more than one idea by construction and must be split.
- **No deliberate contrast in length.** A short sentence placed after a long one reads as a rhetorical device. If a fact deserves its own sentence, give it a normal-length one.
- **Subject and verb in the first eight words.** Long adverbial lead-ins ("Comparing idealised pure dwelling types, ...") are acceptable once per paragraph and never twice in succession.
- **No em-dashes.** Use a comma, a parenthesis, or a new sentence. This is absolute, in prose, captions, and site copy.
- **Semicolons join two clauses at most**, and the second must not be a conclusion drawn from the first. A semicolon followed by "therefore" is always a rewrite.
- **Colons introduce a list or a definition**, not a payoff.
- **Negation states a fact, not an emphasis.** One negated construction per paragraph. "Technology reduces consumption without closing either gap" is a finding; three negations in four sentences is a drumbeat.
- **Parentheses hold references, units and CIs.** Do not use them for asides that carry argument.

## 3. Appositives

An appositive is a noun phrase followed by a comma and a restatement of that noun phrase: "through their consumption, the energy that homes and vehicles use". It is the most common fault in this manuscript and gets its own section.

- **Budget: one per abstract, one per paragraph elsewhere, zero in the last sentence of any section.**
- **Prefer the gloss to the head noun.** In most cases the restatement is the real content and the head noun is vague. Delete the head noun and promote the gloss.
- **Never stack two on one head noun.** "a single evaluation, the everyday access per unit of energy spent, an energy productivity for places analogous to that of economies" is three attempts at one definition.
- **A caveat is not an appositive.** "the most favourable case for the deployment pathway" is a real methodological concession and deserves its own clause or sentence, where a reader will register it.

The test: read the sentence with the material between the commas removed. If it still says something, the appositive is decoration. If it says nothing, the head noun is the decoration.

## 4. Terminology

One term per concept, repeated without variation. Synonym rotation is a literary habit that costs precision in a paper where several nearby quantities have distinct definitions.

| Concept | Use | Do not use |
|---|---|---|
| The access measure | access | reach, everyday reach, the count |
| The planning good itself | proximity | (correct in policy contexts; do not use it for the measured count) |
| Access per unit energy | the rate | ratio, productivity (except where energy productivity is defined as the concept) |
| The two quantities and the rate together | access and energy, and the rate between them; NEPI | axis, the two axes, the two-axis evaluation (geometry metaphor for two measured quantities); the measure, the combined measure (collides with policy measures) |
| Compositional contrast | the gap | difference, penalty, disparity |
| Insulation, heat pumps, electrification | measures, mitigation measures | interventions, solutions, demand-side measures (reads as demand response; the paper's contrast is assets vs form) |
| The Jacobs-inspired premise | the energy productivity principle (lowercase; named at first use in the intro) | the Jacobs premise, the conduit principle, her hypothesis |
| Energy at the dwelling | home energy | household energy (that is the equal-household-size view) |
| Home plus travel per dwelling | total energy | household energy |
| Car travel distance band | car catchment | catchment (unqualified), range |
| The status quo scenario | as-lived (manuscript only), status quo (site copy) | as lived, lived experience, current state |
| Dwelling types in labels | Flats, Terraced, Semi-detached, Detached | flat, semi, house types |

Other standing conventions:

- British spelling and British adverb forms throughout.
- Oxford (serial) comma in every list of three or more items.
- No contractions. No first-person singular. "We" is acceptable for method decisions.
- **Hedge budget: "about" once per paragraph.** Repeating the same hedge before every number turns it into a beat and stops signalling anything. Where several approximate values appear together, hedge the set once and give the values plain.
- **"X times the amenities", not "X times more amenities".** The second is ambiguous between the ratio and the increment.
- Deliberately approximate prose values ("about 2,500 miles") are allowed and are listed as static exceptions in the submission checklist. Every other number is a `\nepi` macro.

## 5. The fault catalogue

Each fault is stated with a diagnostic and a fix. Examples are drawn from real drafts of this manuscript.

**F1. The appositive gloss.** See §3.
Before: "Energy and decarbonisation policy evaluate neighbourhoods through their consumption, the energy that homes and vehicles use."
After: "Energy and decarbonisation policy evaluate neighbourhoods through the energy that homes and vehicles use."

**F2. Mirrored restatement.** Two consecutive sentences make the same claim in different words. Diagnostic: the two sentences share a subject and a verb sense.
Standing exception (author's decision, 2026-08-07): the Jacobs conduit paragraph states the principle twice before the rainforest illustration. The repetition is deliberate, so that the reader grasps the magnitude of the claim before the analogy lands. Do not flag or delete it.
Before: "This paper combines the two into a single evaluation, the everyday access a neighbourhood obtains per unit of energy spent. The combined measure captures how effectively a place converts energy into everyday reach, and it expresses proximity in the same unit as retrofit and electrification."
After: "This paper measures the everyday access a neighbourhood obtains per unit of energy spent. The measure expresses access in the same unit as retrofit and electrification, so the three can be compared in one accounting."

**F3. The punch sentence.** A very short sentence following a very long one, used to land a point.
Before: "... and heat pumps widen the ratio. No measure changes the access gap."
After: "Heat pumps widen the energy ratio, and no modelled measure changes the access gap."

**F4. The semicolon ladder.** Three findings chained in one sentence with escalating structure and a concluding "therefore". Fix by splitting into one sentence per finding and deleting the connective.

**F5. Elegant variation.** The same quantity named four ways (access, everyday reach, destinations residents can reach, the count). Fix with §4.

**F6. The hedge beat.** The same hedge word in the same slot before every number. Fix with the hedge budget in §4.

**F7. The meta-move sentence.** A sentence describing what the paper does rather than reporting what it found: "We ask what this evaluation shows for England". Abstracts and results sections report; only the last paragraph of an introduction may describe the paper's structure, and only in one sentence. The same fault includes bare taxonomic announcements ("Three families of checks apply."): fold the classification into a sentence that carries content, or attach the first item with a colon.

**F8. The closing epigram.** A final clause built for emphasis rather than content, usually an appositive, often ending in "itself", "alone" or "the whole point".
Before: "... should place greater emphasis on compact urban form, the lever that acts on the lock-in itself."
After: "... should place greater emphasis on compact urban form."

**F9. The mirrored setup.** Field A does X. Field B does Y instead. This paper combines them. A stock three-beat opening. Keep the contrast if it is real, but state it once and without matched verbs, and do not follow it with a sentence announcing the synthesis.

**F10. Motif repetition.** One idea restated through recurring vocabulary across nearby sentences ("combines the two", "a single evaluation", "the combined measure", "within one accounting"). Choose one phrasing and delete the others.

**F11. Grand framing.** Weight-words applied to ordinary results: structural, decisive, fundamental, profound, the story, the journey, elegant, powerful. A result is described by its size and its confidence interval.

**F12. Coined motifs and sustained metaphor.** A metaphor introduced once and then reused as a name for the finding becomes a claim the data do not support. Analogies are single-use (see §7).

**F13. Tacked-on qualifiers.** "including out-of-sample", "full stop", "period", "and rightly so". Delete on sight.

**F14. Rhetorical inversion.** "That the gap survives full electrification is the evidence that ...". Rewrite as a plain subject-verb-object statement of what survived and what that implies.

**F15. Casual register.** Conversational idiom in place of formal statement: "for decades" (use "has long"), "if anything", "on its own" (use "alone"), "does not move" (use "remains unchanged"), the pronoun "one" standing for a technical noun ("a delta-method one"), "however often". The test: would the phrase survive in a methods section? If not, it does not belong in the abstract or discussion either.

**F16. Mirrored noun pairs.** Two coordinated concrete pairs balanced across a verb: "the walls and distances that set demand outlast the boilers and cars that policy replaces". One concrete pair per sentence at most; the second pair converts a claim into an epigram. Related to F8 and F12.

**F17a. Colon chains.** The colon as default connector: "X: Y" used sentence after sentence to attach an elaboration, definition, example or payoff. Budget: at most one colon per paragraph, and only for a genuine enumeration or definition; never in consecutive sentences. The fix is usually a plain conjunction ("for", "because", "so"), a relative clause, or a second sentence.

**F17. Absolutist claims about literatures and instruments.** "Neither field currently answers", "never entered", "not scored by any existing instrument", "at all". These invite a counterexample the paper then has to defend. Scope them: "ordinarily", "typically", "chiefly", "in current use", "no established". The exception is a claim true by construction ("none of these technologies alters the distance between dwellings and destinations"), which stays absolute because hedging it would misstate the model. Do not over-correct into vagueness; the scoped claim must still commit.

## 6. Section-level construction

- **Abstract.** Context in one or two sentences, the measure in one, the data and scope in one, three or four result sentences, one implication sentence. No sentence describing the paper's own novelty. No appositive after the first. Target 200 to 250 words with sentence lengths inside the §2 band.
- **Introduction.** Ends with the contribution stated as claims, not as a tour of the sections.
- **Results.** One finding per paragraph, opening with the finding. Numbers carry units and CIs or point to the table that does. No interpretation beyond the comparison being made.
- **Discussion.** Every claim traceable to a numbered result or a citation. Limitations stated in their own sentences with the direction of the likely bias.
- **Methods.** Sufficient for independent replication. Imperative or passive is acceptable here; brevity is not a virtue at the expense of a reproducible step.
- **Captions.** First sentence states what is plotted. Second states what it shows. No third sentence unless a definition is needed. Caption text does not argue.

## 7. Analogies

- One analogy per document, used once, never revived as shorthand for the result.
- Standing exception (author's decision, 2026-08-07): the rainforest/desert analogy returns once in the Discussion's Jacobs paragraph, carrying the measured numbers.
- The analogy must map onto the measured quantity in the same direction. The standing example: the "equivalent amount of energy" belongs on the desert, not on the rainforest.
- If an analogy needs a sentence of explanation, it is not doing useful work.

## 8. Figures and labels

Full conventions are in [figure_notes.md](figure_notes.md). The prose-facing rules:

- Identical quantities carry identical axis labels across every figure.
- Tick text is styled as a label, not as data.
- Log axes are marked with a short "(log)", not a phrase.
- Dwelling types are always Flats, Terraced, Semi-detached, Detached, in that order.
- Text in a figure follows every rule in this guide, at greater strictness, because it cannot be revised in proof.

## 8a. The gate (run on every sentence before it is shown)

The faults in §5 return in rotation because each correction removes one device and leaves the impulse intact: an em-dash becomes a colon, a colon becomes a fragment, a fragment becomes a fronted count. The constraint that stops the rotation is on content, not form. **Each sentence names an actor that can literally act and reports one measured fact or one standing state. A sentence carrying emphasis instead of a fact is deleted, not rewritten.** Ten yes/no checks:

1. **Subject.** Is the grammatical subject a discourse object (paper, study, results, analysis, design, comparison, test, argument, strand, literature) governing an agentive verb? Licensed actors: people, residents, neighbourhoods, dwellings, destinations, records, measurements, policy instruments.
2. **Given-new.** Does the first noun phrase name something absent from the preceding sentence, or open with a bare cardinal plus a new noun ("Three hazards could mislead...")?
3. **Length.** Under 10 words without being a definition or a scope statement, or over 45 words.
4. **Punctuation.** An em-dash or en-dash; a colon introducing anything but a list or a definition; a semicolon whose second clause concludes from the first.
5. **Symmetry.** Two clauses with matched verbs or matched noun pairs, a paired conditional, "not X but Y", a terminal "rather than", a triad for rhythm.
6. **Closer.** Does the final clause add a fact? Fail if it only re-weights what precedes it.
7. **Literal verbs.** Can the named subject literally perform the verb, and is any metaphor used once rather than as the name of a finding?
8. **Meta-move.** No "we show/ask/argue/present", no "this paper ...". Permitted after "we": measured, compared, fitted, held, recomputed, applied, anchored, read, restricted.
9. **Tense.** Completed work in the past, standing states in the present, never both in one clause.
10. **Stub openers.** A one-sentence opener carrying no context or rationale is a headline. Fold it into the sentence that carries the evidence.

## 9. The revision pass

Run in order on each section. This is a deletion pass; if a step produces an addition, something is wrong.

1. **Count the appositives.** Find every comma followed by "the", "a" or "an", and check whether the phrase after it restates the noun before it. Enforce §3.
2. **Check for em-dashes.** `grep -n -- "---" latex/main.tex` and the en-dash and em-dash characters. Zero is the only acceptable count.
3. **Measure the sentences.** Any sentence over 45 words is split. Any sentence under 10 words is checked against F3.
4. **List the sentence openings.** Repeated openings mean the paragraph has one idea spread over several sentences.
5. **Check consecutive sentence pairs for F2.** Same subject and same verb sense means one of them goes.
6. **Count "about"**, and each other hedge, per paragraph.
7. **Check the terminology table.** One term per concept, no variation.
8. **Check the last sentence of every section against F8.** Sections end on their last fact.
9. **Search the banned vocabulary**: crucially, importantly, notably, it is worth noting, in other words, underscores, highlights, sheds light, the very, not only, not just, arguably, quite simply.
10. **Read the section aloud.** Any place where the voice wants to rise or pause for effect is a place where the sentence was built for rhythm.

Mechanical checks that catch most of the catalogue in one pass:

```bash
grep -n "—\|–" paper/latex/main.tex          # F: em-dash and en-dash asides
grep -n "not only\|not just\|itself\.\|alone\." paper/latex/main.tex
grep -n "crucially\|importantly\|notably\|underscore\|highlight" paper/latex/main.tex
grep -c "about " paper/latex/main.tex        # hedge budget
```

## 10. What this guide does not cover

Argument, evidence and scope. A section can pass every rule here and still overclaim. The prose rules exist so that the claim in a sentence is visible, which is what makes an overclaim reviewable. When a rule in this guide would obscure a claim, the claim wins and the exception is recorded here.
