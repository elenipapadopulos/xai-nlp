import random
import pandas as pd

SEED = 43

SENTENCE_BASES = [
    ("film",     "The movie was"),
    ("film",     "The film was"),
    ("film",     "The plot was"),
    ("film",     "The acting was"),
    ("film",     "The ending was"),
    ("film",     "The soundtrack was"),
    ("film",     "The script was"),
    ("film",     "The cinematography was"),
    ("film",     "The cast was"),
    ("film",     "The direction was"),
    ("food",     "The pasta was"),
    ("food",     "The soup was"),
    ("food",     "The bread was"),
    ("food",     "The dessert was"),
    ("food",     "The steak was"),
    ("food",     "The salad was"),
    ("food",     "The coffee was"),
    ("food",     "The sauce was"),
    ("food",     "The pizza was"),
    ("food",     "The meal was"),
    ("people",   "The teacher was"),
    ("people",   "The doctor was"),
    ("people",   "The colleague was"),
    ("people",   "The manager was"),
    ("people",   "The guide was"),
    ("people",   "The host was"),
    ("people",   "The performer was"),
    ("people",   "The speaker was"),
    ("people",   "The coach was"),
    ("people",   "The chef was"),
    ("places",   "The beach was"),
    ("places",   "The park was"),
    ("places",   "The museum was"),
    ("places",   "The restaurant was"),
    ("places",   "The hotel was"),
    ("places",   "The city was"),
    ("places",   "The garden was"),
    ("places",   "The view was"),
    ("places",   "The village was"),
    ("places",   "The market was"),
    ("products", "The laptop was"),
    ("products", "The phone was"),
    ("products", "The camera was"),
    ("products", "The headphones were"),
    ("products", "The keyboard was"),
    ("products", "The watch was"),
    ("products", "The bag was"),
    ("products", "The software was"),
    ("products", "The chair was"),
    ("products", "The blender was"),
    ("film",     "The dialogue was"),
    ("film",     "The pacing was"),
    ("film",     "The visual effects were"),
    ("film",     "The story was"),
    ("food",     "The wine was"),
    ("food",     "The appetizer was"),
    ("food",     "The portion was"),
    ("food",     "The flavor was"),
    ("people",   "The waiter was"),
    ("people",   "The receptionist was"),
    ("people",   "The instructor was"),
    ("people",   "The assistant was"),
    ("places",   "The neighborhood was"),
    ("places",   "The atmosphere was"),
    ("places",   "The venue was"),
    ("places",   "The landscape was"),
    ("products", "The battery was"),
    ("products", "The screen was"),
    ("products", "The design was"),
    ("products", "The packaging was"),
]

PLAIN_TEMPLATES = [
    # direct
    "{base} {NOT_adj}{adj_phrase}",
    "Honestly, {base_lc} {NOT_adj}{adj_phrase}",
    "Frankly, {base_lc} {NOT_adj}{adj_phrase}",
    "Surprisingly, {base_lc} {NOT_adj}{adj_phrase}",
    "Admittedly, {base_lc} {NOT_adj}{adj_phrase}",
    "Clearly, {base_lc} {NOT_adj}{adj_phrase}",
    "Undeniably, {base_lc} {NOT_adj}{adj_phrase}",
    "I have to say {base_lc} {NOT_adj}{adj_phrase}",
    "I must admit {base_lc} {NOT_adj}{adj_phrase}",
    "If I am honest, {base_lc} {NOT_adj}{adj_phrase}",
    "I think {base_lc} {NOT_adj}{adj_phrase}",
    "I feel like {base_lc} {NOT_adj}{adj_phrase}",
    "I would argue {base_lc} {NOT_adj}{adj_phrase}",
    "In the end, {base_lc} {NOT_adj}{adj_phrase}",
    "Overall, {base_lc} {NOT_adj}{adj_phrase}",
    "At the end of the day, {base_lc} {NOT_adj}{adj_phrase}",
    "Looking back, {base_lc} {NOT_adj}{adj_phrase}",
    "All things considered, {base_lc} {NOT_adj}{adj_phrase}",
    "From the start, {base_lc} {NOT_adj}{adj_phrase}",
    "Right from the beginning, {base_lc} {NOT_adj}{adj_phrase}",
    "I remember thinking that {base_lc} {NOT_adj}{adj_phrase}",
    "To my surprise, {base_lc} {NOT_adj}{adj_phrase}",
    "Much to my surprise, {base_lc} {NOT_adj}{adj_phrase}",
    "Everyone agreed that {base_lc} {NOT_adj}{adj_phrase}",
    "People kept saying {base_lc} {NOT_adj}{adj_phrase}",
    "I heard they claimed that {base_lc} {NOT_adj}{adj_phrase}",
    "Nobody could deny that {base_lc} {NOT_adj}{adj_phrase}",
    "Without a doubt, {base_lc} {NOT_adj}{adj_phrase}",
    "There is no denying that {base_lc} {NOT_adj}{adj_phrase}",
    "No matter how you look at it, {base_lc} {NOT_adj}{adj_phrase}",
    "At best, {base_lc} {NOT_adj}{adj_phrase}",
    "At worst, {base_lc} {NOT_adj}{adj_phrase}",
    "In my experience, {base_lc} {NOT_adj}{adj_phrase}",
    "From my perspective, {base_lc} {NOT_adj}{adj_phrase}",
    "As far as I could tell, {base_lc} {NOT_adj}{adj_phrase}",
    "Compared to what I expected, {base_lc} {NOT_adj}{adj_phrase}",
    "Let us be clear, {base_lc} {NOT_adj}{adj_phrase}",
    "Make no mistake, {base_lc} {NOT_adj}{adj_phrase}",
    "To put it mildly, {base_lc} {NOT_adj}{adj_phrase}",
    "To say the least, {base_lc} {NOT_adj}{adj_phrase}",
    "If anything, {base_lc} {NOT_adj}{adj_phrase}",
    "Even so, {base_lc} {NOT_adj}{adj_phrase}",
    "That said, {base_lc} {NOT_adj}{adj_phrase}",
    "Who would have thought that {base_lc} {NOT_adj}{adj_phrase}",
    "Having given it a fair chance, {base_lc} {NOT_adj}{adj_phrase}",
    "Looking at it carefully, {base_lc} {NOT_adj}{adj_phrase}",
]

ADJECTIVES = { # to update
    "film": {
        "positive": ["captivating", "brilliant", "gripping", "stunning", "masterful",
                     "compelling", "moving", "unforgettable", "thought-provoking", "mesmerizing"],
        "negative": ["predictable", "boring", "derivative", "forgettable", "tedious",
                     "confusing", "pretentious", "shallow", "disappointing", "overlong"],
    },
    "food": {
        "positive": ["delicious", "flavorful", "fresh", "tender", "crispy",
                     "rich", "heavenly", "savory", "mouthwatering", "perfectly seasoned"],
        "negative": ["bland", "tasteless", "stale", "soggy", "overcooked",
                     "greasy", "cold", "salty", "undercooked", "disappointing"],
    },
    "people": {
        "positive": ["inspiring", "caring", "knowledgeable", "supportive", "talented",
                     "engaging", "welcoming", "motivating", "helpful", "creative"],
        "negative": ["rude", "incompetent", "unhelpful", "dismissive", "arrogant",
                     "condescending", "unprofessional", "unreliable", "careless", "boring"],
    },
    "places": {
        "positive": ["breathtaking", "peaceful", "vibrant", "charming", "serene",
                     "magnificent", "cozy", "lively", "fascinating", "beautiful"],
        "negative": ["overcrowded", "dirty", "noisy", "overpriced", "run-down",
                     "disappointing", "chaotic", "unsafe", "dull", "ugly"],
    },
    "products": {
        "positive": ["reliable", "precise", "durable", "intuitive", "comfortable",
                     "responsive", "elegant", "powerful", "sturdy", "impressive"],
        "negative": ["broken", "slow", "unreliable", "fragile", "defective",
                     "glitchy", "outdated", "flimsy", "uncomfortable", "overpriced"],
    },
}



AUX_TEMPLATES = [
    "I could not wait to leave because {base_lc} {NOT_adj}{adj_phrase}",
    "I would not go back: {base_lc} {NOT_adj}{adj_phrase}",
    "I could not believe that {base_lc} {NOT_adj}{adj_phrase}",
    "I could not stop thinking about how {base_lc} {NOT_adj}{adj_phrase}",
    "I could not help but notice that {base_lc} {NOT_adj}{adj_phrase}",
    "I would not say that {base_lc} {NOT_adj}{adj_phrase}",
    "You could not deny that {base_lc} {NOT_adj}{adj_phrase}",
    "It was not clear to me that {base_lc} {NOT_adj}{adj_phrase}",
    "It was not obvious that {base_lc} {NOT_adj}{adj_phrase}",
    "Nobody could not see that {base_lc} {NOT_adj}{adj_phrase}",
    "Everyone could not tell that {base_lc} {NOT_adj}{adj_phrase}",
    "I will not forget that {base_lc} {NOT_adj}{adj_phrase}"
]

def compose_not(n_not):
    """compose_not(2) -> "not not" """
    return " ".join(["not"] * n_not) if n_not > 0 else ""


def build_not_phrase(base, adjs, n_not, contraction, not_at, aux_not, rng):

    """
    "base": base sentence (e.g. "The movie was")
    "adjs": sampled adjective
    "n_not": number of "not" to stack (e.g. "not not good")
    "contraction": whether to contract "not" into the verb (e.g. "wasn't good")
    "not_at": whether to prepend "NOT" at the beginning of the sentence (e.g. "NOT The movie was good")
    "aux_not": whether to use an auxiliary verb template (e.g. "I could not wait to leave because the movie was not good")
    """

    adj_phrase = adjs[0]

    if contraction:
        # (the relevant) NOT contracts into the verb: "was" -> "wasn't"
        # remaining (n_not - 1) NOTs stack before the adjective
        contracted_base = base.replace(" was", " wasn't").replace(" were", " weren't")
        NOT_adj         = compose_not(n_not - 1) + " " if n_not > 1 else ""
        sentence        = f"{contracted_base} {NOT_adj}{adj_phrase}"
    else:
        NOT_adj = compose_not(n_not) + " " if n_not > 0 else ""
        if aux_not:
            t_i      = rng.randrange(len(AUX_TEMPLATES))
            sentence = AUX_TEMPLATES[t_i].format(
                base_lc    = base[0].lower() + base[1:],
                adj_phrase = adj_phrase,
                NOT_adj    = NOT_adj
            )
        else:
            t_i      = rng.randrange(len(PLAIN_TEMPLATES))
            sentence = PLAIN_TEMPLATES[t_i].format(
                base    = base,
                base_lc = base[0].lower() + base[1:],
                adj_phrase = adj_phrase,
                NOT_adj    = NOT_adj,
            )
            # sentence = f"{base} {NOT_adj}{adj_phrase}"


    if not_at:
        sentence = "NOT " + sentence

    return sentence


def generate_not_sentences(
    
    adj_sentiment    = "positive", 
    n_not        = 1,
    contraction  = False,
    not_at       = False,
    aux_not      = False,
    seed         = SEED,
):
    
    """
    Generate sentences with 'NOT' preposition.
    "adj_sentiment": positive, negative or both
    "n_not": number of "not" to stack (e.g. "not not good")
    "contraction": whether to contract "not" into the verb (e.g. "wasn't good")
    "not_at": whether to prepend "not" at the beginning of the sentence (e.g. "not The movie was good")
    "aux_not": whether to use an auxiliary verb template (e.g. "I could not wait to leave because the movie was not good")
    """

    rng     = random.Random(seed)

    records = []

    for i, (domain, base) in enumerate(SENTENCE_BASES):
        adj_sent = rng.choice(["positive", "negative"]) if adj_sentiment == "both" else adj_sentiment
        pool = ADJECTIVES[domain][adj_sent]
        adjs = rng.sample(pool, 1)
        total_not = n_not + (1 if not_at else 0)
        label = adj_sent if total_not % 2 == 0 else ("negative" if adj_sent == "positive" else "positive")
        records.append({
            "id":        i + 1,
            "domain":    domain,
            "base":      base,
            "adjs":      adjs[0],
            "adj_sentiment": adj_sent,
            "n_not":     n_not,
            "contraction": contraction,
            "not_at":      not_at,
            "aux_not":   aux_not,
            "sentence":  build_not_phrase(base, adjs, n_not, contraction, not_at, aux_not, rng),
            "label":        label,

        })

    return pd.DataFrame(records)

