"""Fresh robustness QA suite for DocuBot.

The expectation metadata in this file is used *after* AnswerService returns.
It is never included in the question/prompt/retrieval request.

The suite intentionally uses multiple paraphrases per semantic case so repeated
QA cycles can exercise different wording instead of replaying one exact prompt.
"""
from __future__ import annotations

NO_RESULT_MESSAGE = "Information not found in company knowledge base."


def _case(case_id, category, variants, required_groups=(), forbidden_terms=(), *,
          exact_answer="", answer_starts=(), min_bullets=0, max_bullets=0,
          expect_fallback=False, quick=False, temporal_order=()):
    return {
        "case_id": case_id,
        "category": category,
        "variants": list(variants),
        "required_groups": [list(group) for group in required_groups],
        "forbidden_terms": list(forbidden_terms),
        "exact_answer": exact_answer,
        "answer_starts": list(answer_starts),
        "min_bullets": int(min_bullets),
        "max_bullets": int(max_bullets),
        "expect_fallback": bool(expect_fallback),
        "quick": bool(quick),
        "temporal_order": list(temporal_order),
    }


STANDALONE_CASES = [
    _case("R-001", "Identity paraphrase", [
        "Give me a short description of Apolinario Mabini.",
        "In one concise answer, who was Apolinario Mabini?",
        "How is Apolinario Mabini described in the stored documents?",
    ], [["Apolinario Mabini", "Mabini"], ["Prime Minister", "revolutionary leader", "statesman"]], quick=True),
    _case("R-002", "Position relation", [
        "Which national office is Mabini noted for serving in?",
        "What government office was held by Apolinario Mabini?",
        "Name the major government position associated with Mabini.",
    ], [["Prime Minister"]], quick=True),
    _case("R-003", "Role explanation", [
        "What made Mabini important to the revolutionary government?",
        "Describe Mabini's contribution to the revolutionary government.",
        "Why was Mabini significant in the Philippine revolutionary government?",
    ], [["adviser", "Prime Minister", "government"], ["revolution", "Republic", "constitutional", "independence"]]),
    _case("R-004", "Identity paraphrase", [
        "Describe Andres Bonifacio in a sentence.",
        "How do the documents identify Andres Bonifacio?",
        "Give a brief profile of Andres Bonifacio.",
    ], [["Andres Bonifacio", "Andrés Bonifacio", "Bonifacio"], ["revolutionary"]], quick=True),
    _case("R-005", "Organization relation", [
        "Which revolutionary society did Bonifacio help establish?",
        "Name the movement that counted Bonifacio among its founders.",
        "Bonifacio helped found which organization?",
    ], [["Katipunan"]], quick=True),
    _case("R-006", "Leadership title", [
        "What leadership title did Bonifacio hold in the Katipunan?",
        "Which top Katipunan position is associated with Bonifacio?",
        "What office did Bonifacio occupy within the Katipunan?",
    ], [["Supremo", "President", "Pangulo"]]),
    _case("R-007", "Purpose paraphrase", [
        "What goal was the Katipunan organized to pursue?",
        "What outcome did the Katipunan primarily seek?",
        "What was the central objective behind the Katipunan?",
    ], [["independence", "independent", "armed revolution against Spain", "revolution against Spain"]], quick=True),
    _case("R-008", "Tagalog purpose", [
        "Ano ang nais makamit ng Katipunan?",
        "Para sa anong pangunahing adhikain nabuo ang Katipunan?",
        "Ano ang pangunahing mithiin ng Katipunan?",
    ], [["independence", "kalayaan", "malaya"]], quick=True),
    _case("R-009", "Founders list", [
        "Name the people identified as founders of the Katipunan.",
        "Sino-sino ang mga binanggit na nagtatag ng Katipunan?",
        "Which Filipino nationalists are listed as the Katipunan's founders?",
    ], [["Bonifacio"], ["Arellano"], ["Plata"], ["Diwa"], ["Dizon"], ["Diaz", "Díaz"]], min_bullets=4),
    _case("R-010", "Founding date", [
        "On what date was the Katipunan founded?",
        "Kailan itinatag ang Katipunan?",
        "Give the founding date of the Katipunan.",
    ], [["July 7, 1892", "7 July 1892", "Hulyo 7, 1892"]], quick=True),
    _case("R-011", "Treaty date", [
        "Give the signing date of the 1898 Treaty of Paris.",
        "Anong araw nilagdaan ang Treaty of Paris ng 1898?",
        "When was the 1898 peace treaty between Spain and the United States signed?",
    ], [["December 10, 1898", "10 December 1898", "Disyembre 10, 1898"]], quick=True),
    _case("R-012", "Treaty parties", [
        "Which two countries were parties to the Treaty of Paris of 1898?",
        "Sino ang dalawang bansang lumagda sa Treaty of Paris noong 1898?",
        "Name the parties to the 1898 Treaty of Paris.",
    ], [["Spain", "España", "Espanya", "Kaharian ng Espanya", "Kingdom of Spain"], ["United States", "Estados Unidos"]]),
    _case("R-013", "Treaty territories", [
        "Which territories did Spain give up under the 1898 Treaty of Paris?",
        "List the territories Spain relinquished in the Treaty of Paris.",
        "Anong mga teritoryo ang binitiwan ng Spain sa Treaty of Paris?",
    ], [["Cuba"], ["Puerto Rico"], ["Guam"], ["Philippines", "Philippine Islands"]], min_bullets=3),
    _case("R-014", "Treaty effect", [
        "What conflict did the Treaty of Paris of 1898 officially end?",
        "Which war was concluded by the 1898 Treaty of Paris?",
        "Anong digmaan ang opisyal na tinapos ng Treaty of Paris ng 1898?",
    ], [["Spanish-American War", "Spanish–American War"]]),
    _case("R-015", "Date comparison", [
        "Was the Treaty of Paris signed before or after the June 1898 Philippine declaration of independence?",
        "Which came later: the June 1898 declaration of Philippine independence or the Treaty of Paris signing?",
        "Nauna ba ang deklarasyon ng kalayaan noong Hunyo 1898 kaysa pagpirma sa Treaty of Paris?",
    ], [
        ["June 12, 1898", "12 June 1898", "Hunyo 12, 1898"],
        ["December 10, 1898", "10 December 1898", "Disyembre 10, 1898"],
        ["after", "later", "nauna", "sumunod", "first"],
    ], temporal_order=("June 12, 1898", "December 10, 1898")),
    _case("R-016", "Leave quantity", [
        "What is the yearly sick-leave allotment in the policy?",
        "State the annual number of sick-leave days.",
        "Ilang araw ang taunang sick leave ayon sa policy?",
    ], [["17"]], quick=True),
    _case("R-017", "Leave quantity", [
        "What is the yearly vacation-leave allotment in the policy?",
        "State the annual number of vacation-leave days.",
        "Ilang araw ang taunang vacation leave ayon sa policy?",
    ], [["47"]]),
    _case("R-018", "Approver paraphrase", [
        "Whose approval is required for a leave request?",
        "Which role must approve employee leave requests?",
        "Kanino dapat manggaling ang approval ng leave request?",
    ], [["Manager"]], quick=True),
    _case("R-019", "Approver Tagalog", [
        "Sino ang kailangang magbigay ng pahintulot sa leave request?",
        "Kanino dadaan ang leave request para maaprubahan?",
        "Anong role ang may approval authority sa leave request?",
    ], [["Manager"]], quick=True),
    _case("R-020", "Eligibility", [
        "Which employees qualify for the leave benefits in the policy?",
        "Who is covered by the listed leave entitlements?",
        "Sino ang eligible sa mga leave na nakasaad sa policy?",
    ], [["regular employee", "regular employees"]], quick=True),
    _case("R-021", "Leave comparison", [
        "Compare the annual vacation and sick leave allowances.",
        "How do the yearly VL and SL amounts differ?",
        "Ihambing ang taunang vacation leave at sick leave credits.",
    ], [["47"], ["17"]], quick=True),
    _case("R-022", "Abbreviation robustness", [
        "How much VL and SL does the policy provide each year?",
        "Ilang VL at SL credits ang nakalagay kada taon?",
        "Give the annual VL and SL figures from the leave policy.",
    ], [["47"], ["17"]]),
    _case("R-023", "Structured statement", [
        "State the requirement written in MISRA Rule 1.2.",
        "What requirement is imposed by Rule 1.2?",
        "Give the exact rule statement for MISRA 1.2.",
    ], [["Language extensions should not be used", "language extensions"]], quick=True),
    _case("R-024", "Structured category", [
        "How is MISRA Rule 1.2 categorized?",
        "What category label is assigned to Rule 1.2?",
        "Mandatory, required, or advisory: which one is Rule 1.2?",
    ], [["Advisory"]], quick=True),
    _case("R-025", "Structured applicability", [
        "Which C standards are covered by Rule 1.2?",
        "What C versions does MISRA Rule 1.2 apply to?",
        "Give the applicability versions for Rule 1.2.",
    ], [["C90"], ["C99"]], quick=True),
    _case("R-026", "Structured analysis", [
        "What analysis classification is attached to Rule 1.2?",
        "Give the Analysis value for MISRA Rule 1.2.",
        "How is Rule 1.2 classified under Analysis?",
    ], [["Undecidable"], ["Single Translation Unit"]]),
    _case("R-027", "Structured rationale", [
        "Why does MISRA advise against relying on language extensions?",
        "What portability concern is behind Rule 1.2?",
        "Explain the reason for discouraging language extensions in Rule 1.2.",
    ], [["portable", "portability"], ["extension"]]),
    _case("R-028", "Structured explanation", [
        "Explain MISRA Rule 1.2 in plain language.",
        "Give a practical explanation of what Rule 1.2 means.",
        "Ipaliwanag sa simpleng paraan ang MISRA Rule 1.2.",
    ], [["language extension", "extensions"], ["portability", "portable"]]),
    _case("R-029", "Section overview", [
        "Give a concise overview of MISRA Section 6.",
        "What topics does Section 6 introduce?",
        "Summarize the purpose of Section 6 without listing every detail.",
    ], [["guideline"], ["classification", "category", "categories"]], max_bullets=12),
    _case("R-030", "Section detail", [
        "How does Section 6 distinguish a rule from a directive?",
        "According to Section 6, what is the difference between rules and directives?",
        "Ano ang pagkakaiba ng rule at directive ayon sa Section 6?",
    ], [["rule"], ["directive"]], quick=True),
    _case("R-031", "Section categories", [
        "Which three guideline categories are defined in Section 6?",
        "List the categories assigned to MISRA guidelines in Section 6.",
        "Ano ang tatlong guideline categories sa Section 6?",
    ], [["mandatory"], ["required"], ["advisory"]], min_bullets=2),
    _case("R-032", "Section decidability", [
        "What does Section 6 say about decidable and undecidable rules?",
        "Explain the decidability classification described in Section 6.",
        "How does Section 6 treat rules that cannot always be decided statically?",
    ], [["decidable"], ["undecidable"]]),
    _case("R-033", "Generated code", [
        "What guidance does Section 6 give for automatically generated code?",
        "How are MISRA guidelines applied to auto-generated source code?",
        "Ano ang sinasabi ng Section 6 tungkol sa automatically generated code?",
    ], [["generated", "automatically generated"], ["guideline", "compliance", "code"]]),
    _case("R-034", "Aguinaldo identity", [
        "Give a brief description of Emilio Aguinaldo.",
        "How is Emilio Aguinaldo identified in the stored material?",
        "Who was Emilio Aguinaldo, in brief?",
    ], [["Emilio Aguinaldo", "Aguinaldo"], ["President", "revolutionary"]]),
    _case("R-035", "Aguinaldo role", [
        "Which presidency is Emilio Aguinaldo associated with?",
        "What national leadership office did Aguinaldo hold?",
        "Anong pangunahing posisyon sa pamahalaan ang hinawakan ni Emilio Aguinaldo?",
    ], [["President"]]),
    _case("R-036", "Rizal identity", [
        "Give a short profile of Jose Rizal from the knowledge base.",
        "How do the documents describe Jose Rizal?",
        "In brief, what is Jose Rizal known for?",
    ], [["Rizal"], ["writer", "nationalist", "hero", "polymath"]]),
    _case("R-037", "Rizal-Katipunan relation", [
        "Why did Katipunan representatives seek Rizal's advice?",
        "What issue did the Katipunan want to consult Rizal about?",
        "Bakit hinanap ng Katipunan ang payo ni Rizal?",
    ], [["revolution", "revolt", "Spain"], ["advice", "support", "consult"]]),
    _case("R-038", "Independence date", [
        "On what date did Aguinaldo issue the Philippine Declaration of Independence?",
        "Give the 1898 declaration date of Philippine independence.",
        "Kailan inilabas ni Aguinaldo ang deklarasyon ng kalayaan ng Pilipinas?",
    ], [["June 12, 1898", "12 June 1898", "Hunyo 12, 1898"]]),
    _case("R-039", "Katipunan abbreviation", [
        "What abbreviation is used for the Katipunan?",
        "How is the Katipunan commonly abbreviated in the document?",
        "Ano ang abbreviation ng Katipunan?",
    ], [["KKK"]]),
    _case("R-040", "Katipunan leadership", [
        "Who is listed as president of the Katipunan in the stored material?",
        "Name the person shown as Katipunan president.",
        "Sino ang nakalistang pangulo ng Katipunan?",
    ], [["Bonifacio"]]),
    _case("R-041", "Negative knowledge", [
        "What parking reimbursement ceiling is provided for employees?",
        "How much may an employee claim for parking expenses?",
        "Magkano ang maximum parking reimbursement ng employee?",
    ], expect_fallback=True, quick=True),
    _case("R-042", "Negative knowledge", [
        "How many birthday-leave days are provided each year?",
        "What is the annual birthday leave entitlement?",
        "Ilang araw ang birthday leave kada taon?",
    ], expect_fallback=True),
    _case("R-043", "Negative knowledge", [
        "What monthly mobile-phone allowance does the company provide?",
        "State the employee cellphone allowance amount.",
        "Magkano ang monthly mobile phone allowance?",
    ], expect_fallback=True, quick=True),
    _case("R-044", "Negative knowledge", [
        "Who authorizes overseas travel reimbursement claims?",
        "Which role approves international travel reimbursements?",
        "Sino ang approver ng overseas travel reimbursement?",
    ], expect_fallback=True),
    _case("R-045", "Negative knowledge", [
        "What is the daily meal subsidy amount for employees?",
        "How much meal allowance is provided per working day?",
        "Magkano ang daily meal subsidy?",
    ], expect_fallback=True),
    _case("R-046", "Negative knowledge", [
        "What is the employee internet reimbursement cap?",
        "State the maximum internet allowance reimbursable each month.",
        "Ano ang monthly internet reimbursement limit?",
    ], expect_fallback=True),
    _case("R-047", "Cross-document event", [
        "What happened to the Philippines under the Treaty of Paris of 1898?",
        "How did the 1898 Treaty of Paris affect the Philippines?",
        "Ano ang naging epekto ng Treaty of Paris sa Pilipinas?",
    ], [["United States"], ["Philippines", "Philippine"]]),
    _case("R-048", "Direct detail", [
        "Where was the Treaty of Paris of 1898 signed?",
        "Name the signing location of the 1898 Treaty of Paris.",
        "Saang lugar nilagdaan ang Treaty of Paris noong 1898?",
    ], [["Paris", "France"]]),
    _case("R-049", "Direct detail", [
        "When did the Treaty of Paris of 1898 become effective?",
        "Give the effective date of the 1898 Treaty of Paris.",
        "Kailan naging effective ang Treaty of Paris ng 1898?",
    ], [["April 11, 1899", "11 April 1899", "Abril 11, 1899"]]),
    _case("R-050", "Multi-fact policy", [
        "For a regular employee, summarize the annual VL, annual SL, and required leave approver.",
        "Give the leave policy's VL days, SL days, and approval role for a regular employee.",
        "Para sa regular employee, ilista ang VL, SL, at sino ang approver ng leave.",
    ], [["47"], ["17"], ["Manager"]], min_bullets=2, quick=True),
]


def _turn(turn_id, variants, required_groups=(), *, expect_fallback=False):
    return {
        "turn_id": turn_id,
        "variants": list(variants),
        "required_groups": [list(group) for group in required_groups],
        "expect_fallback": bool(expect_fallback),
    }


CONVERSATION_CHAINS = [
    {"chain_id": "C-A", "name": "Mabini pronoun recovery", "quick": True, "turns": [
        _turn("A1", ["Give me a short profile of Apolinario Mabini.", "Briefly describe Apolinario Mabini."], [["Mabini"]]),
        _turn("A2", ["Which major office did he hold?", "What national position did he serve in?"], [["Prime Minister"]]),
        _turn("A3", ["Why was that role significant?", "Why did that position matter in his historical role?"], [["government", "Republic", "revolution", "independence"]]),
        _turn("A4", ["What other government responsibility is mentioned for him?", "What additional government role is listed for him?"], [["Foreign Relations", "Secretary", "adviser"]]),
    ]},
    {"chain_id": "C-B", "name": "Bonifacio organization follow-up", "quick": True, "turns": [
        _turn("B1", ["Describe Andres Bonifacio briefly.", "Give a concise profile of Andres Bonifacio."], [["Bonifacio"]]),
        _turn("B2", ["Which organization was he among the founders of?", "What revolutionary group did he help establish?"], [["Katipunan"]]),
        _turn("B3", ["What was its main objective?", "What did that organization primarily seek?"], [["independence", "independent"]]),
        _turn("B4", ["When was it founded?", "What was its founding date?"], [["July 7, 1892", "7 July 1892"]]),
    ]},
    {"chain_id": "C-C", "name": "Leave policy follow-up", "quick": True, "turns": [
        _turn("C1", ["State the annual sick-leave allowance.", "How much sick leave is provided per year?"], [["17"]]),
        _turn("C2", ["Who must approve it?", "Whose approval does it require?"], [["Manager"]]),
        _turn("C3", ["Who qualifies for these leave benefits?", "Which employees are eligible for those benefits?"], [["regular employee", "regular employees"]]),
        _turn("C4", ["How does the vacation-leave amount compare with it?", "What is the corresponding vacation-leave allowance?"], [["47"]]),
    ]},
    {"chain_id": "C-D", "name": "Structured rule follow-up", "quick": False, "turns": [
        _turn("D1", ["State MISRA Rule 1.2.", "Give the requirement in Rule 1.2."], [["language extension", "extensions"]]),
        _turn("D2", ["What category is it assigned to?", "Which category does that rule have?"], [["Advisory"]]),
        _turn("D3", ["Which C versions does it cover?", "What standards does it apply to?"], [["C90"], ["C99"]]),
        _turn("D4", ["Why does that rule exist?", "What rationale is given for it?"], [["portable", "portability"]]),
    ]},
    {"chain_id": "C-E", "name": "Section 6 follow-up", "quick": False, "turns": [
        _turn("E1", ["Give me an overview of Section 6.", "What is Section 6 mainly about?"], [["guideline"]]),
        _turn("E2", ["How does it distinguish rules from directives?", "What difference does it make between a rule and a directive?"], [["rule"], ["directive"]]),
        _turn("E3", ["What categories does it define?", "Which guideline categories are described there?"], [["mandatory"], ["required"], ["advisory"]]),
        _turn("E4", ["What does it say about decidability?", "How does it discuss decidable and undecidable rules?"], [["decidable"], ["undecidable"]]),
    ]},
    {"chain_id": "C-F", "name": "Tagalog Katipunan follow-up", "quick": False, "turns": [
        _turn("F1", ["Ilarawan nang maikli ang Katipunan.", "Ano ang Katipunan sa maikling paliwanag?"], [["Katipunan"]]),
        _turn("F2", ["Sino ang isa sa mga nagtatag nito?", "Magbigay ng isang founder nito."], [["Bonifacio", "Arellano", "Diwa", "Plata", "Dizon", "Diaz"]]),
        _turn("F3", ["Ano ang pangunahing layunin nito?", "Ano ang gusto nitong makamit?"], [["independence", "kalayaan", "malaya"]]),
        _turn("F4", ["Kailan ito itinatag?", "Ano ang petsa ng pagkakatatag nito?"], [["July 7, 1892", "7 July 1892", "Hulyo 7, 1892"]]),
    ]},
    {"chain_id": "C-G", "name": "Treaty follow-up", "quick": False, "turns": [
        _turn("G1", ["When was the 1898 Treaty of Paris signed?", "Give the signing date of the Treaty of Paris of 1898."], [["December 10, 1898", "10 December 1898"]]),
        _turn("G2", ["Which countries were parties to it?", "Who were the two national parties to that treaty?"], [["Spain"], ["United States"]]),
        _turn("G3", ["What did it mean for the Philippines?", "How did that treaty affect the Philippines?"], [["United States"], ["Philippines", "Philippine"]]),
        _turn("G4", ["When did it become effective?", "What was its effective date?"], [["April 11, 1899", "11 April 1899"]]),
    ]},
]
