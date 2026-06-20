import random
import pandas as pd

SEED = 42

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
    ("film",     "The dialogue was"),
    ("film",     "The pacing was"),
    ("film",     "The visual effects were"),
    ("film",     "The story was"),
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
    ("food",     "The wine was"),
    ("food",     "The appetizer was"),
    ("food",     "The portion was"),
    ("food",     "The flavor was"),
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
    ("people",   "The waiter was"),
    ("people",   "The receptionist was"),
    ("people",   "The instructor was"),
    ("people",   "The assistant was"),
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
    ("places",   "The neighborhood was"),
    ("places",   "The atmosphere was"),
    ("places",   "The venue was"),
    ("places",   "The landscape was"),
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
    ("products", "The battery was"),
    ("products", "The screen was"),
    ("products", "The design was"),
    ("products", "The packaging was"),
]

ADJECTIVES = {
    "film": {
        "positive": ["captivating", "brilliant", "gripping", "stunning", "masterful",
                     "compelling", "moving", "unforgettable", "thought-provoking", "mesmerizing"],
        "negative": ["predictable", "boring", "unoriginal", "forgettable", "tedious",
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
                     "responsive", "sleek", "powerful", "sturdy", "impressive"],
        "negative": ["broken", "slow", "unreliable", "fragile", "defective",
                     "faulty", "outdated", "flimsy", "cheap", "overpriced"],
    },
}

PLAIN_TEMPLATES = [
    "{base} {NOT_adj}{adj}",
    "Honestly, {base_lc} {NOT_adj}{adj}",
    "Frankly, {base_lc} {NOT_adj}{adj}",
    "Surprisingly, {base_lc} {NOT_adj}{adj}",
    "Admittedly, {base_lc} {NOT_adj}{adj}",
    "Clearly, {base_lc} {NOT_adj}{adj}",
    "Undeniably, {base_lc} {NOT_adj}{adj}",
    "I have to say {base_lc} {NOT_adj}{adj}",
    "I must admit {base_lc} {NOT_adj}{adj}",
    "If I am honest, {base_lc} {NOT_adj}{adj}",
    "I think {base_lc} {NOT_adj}{adj}",
    "I feel like {base_lc} {NOT_adj}{adj}",
    "I would argue {base_lc} {NOT_adj}{adj}",
    "In the end, {base_lc} {NOT_adj}{adj}",
    "Overall, {base_lc} {NOT_adj}{adj}",
    "At the end of the day, {base_lc} {NOT_adj}{adj}",
    "Looking back, {base_lc} {NOT_adj}{adj}",
    "All things considered, {base_lc} {NOT_adj}{adj}",
    "From the start, {base_lc} {NOT_adj}{adj}",
    "Right from the beginning, {base_lc} {NOT_adj}{adj}",
    "I remember thinking that {base_lc} {NOT_adj}{adj}",
    "To my surprise, {base_lc} {NOT_adj}{adj}",
    "Much to my surprise, {base_lc} {NOT_adj}{adj}",
    "Everyone agreed that {base_lc} {NOT_adj}{adj}",
    "People kept saying {base_lc} {NOT_adj}{adj}",
    "Nobody could deny that {base_lc} {NOT_adj}{adj}",
    "Without a doubt, {base_lc} {NOT_adj}{adj}",
    "There is no denying that {base_lc} {NOT_adj}{adj}",
    "No matter how you look at it, {base_lc} {NOT_adj}{adj}",
    "At best, {base_lc} {NOT_adj}{adj}",
    "At worst, {base_lc} {NOT_adj}{adj}",
    "In my experience, {base_lc} {NOT_adj}{adj}",
    "From my perspective, {base_lc} {NOT_adj}{adj}",
    "As far as I could tell, {base_lc} {NOT_adj}{adj}",
    "Compared to what I expected, {base_lc} {NOT_adj}{adj}",
    "Let us be clear, {base_lc} {NOT_adj}{adj}",
    "Make no mistake, {base_lc} {NOT_adj}{adj}",
    "To put it mildly, {base_lc} {NOT_adj}{adj}",
    "To say the least, {base_lc} {NOT_adj}{adj}",
    "If anything, {base_lc} {NOT_adj}{adj}",
    "Even so, {base_lc} {NOT_adj}{adj}",
    "That said, {base_lc} {NOT_adj}{adj}",
    "Who would have thought that {base_lc} {NOT_adj}{adj}",
    "Having given it a fair chance, {base_lc} {NOT_adj}{adj}",
    "Looking at it carefully, {base_lc} {NOT_adj}{adj}",
]


AUX_TEMPLATES = [
    "I had not realized that {base_lc} {NOT_adj}{adj}",
    "I could not believe that {base_lc} {NOT_adj}{adj}",
    "I could not stop thinking about how {base_lc} {NOT_adj}{adj}",
    "I could not help but notice that {base_lc} {NOT_adj}{adj}",
    "You could not deny that {base_lc} {NOT_adj}{adj}",
    "It was not clear to me at first that {base_lc} {NOT_adj}{adj}",
    "It was not obvious that {base_lc} {NOT_adj}{adj}",
    "She did not mention that {base_lc} {NOT_adj}{adj}",
    "I am not surprised that {base_lc} {NOT_adj}{adj}",
    "They could not explain why {base_lc} {NOT_adj}{adj}",
    "I had not considered that {base_lc} {NOT_adj}{adj}",
]


# --- helpers -----------------------------------------------------------------

def compose_not(n_not):
    return " ".join(["not"] * n_not) if n_not > 0 else ""


def _build_sentence(base, adj, n_not, contraction, not_at, aux_not, rng):
    if contraction:
        contracted_base = base.replace(" was", " wasn't").replace(" were", " weren't")
        NOT_adj         = compose_not(n_not - 1) + " " if n_not > 1 else ""
        t_i             = rng.randrange(len(PLAIN_TEMPLATES))
        sentence        = PLAIN_TEMPLATES[t_i].format(
            base    = contracted_base,
            base_lc = contracted_base[0].lower() + contracted_base[1:],
            NOT_adj = NOT_adj,
            adj     = adj,
        )
    else:
        NOT_adj  = compose_not(n_not) + " " if n_not > 0 else ""
        if aux_not:
            t_i      = rng.randrange(len(AUX_TEMPLATES))
            sentence = AUX_TEMPLATES[t_i].format(
                base_lc = base[0].lower() + base[1:],
                NOT_adj=NOT_adj, 
                adj=adj)
        else:
            t_i      = rng.randrange(len(PLAIN_TEMPLATES))
            sentence = PLAIN_TEMPLATES[t_i].format(
                base    = base,
                base_lc = base[0].lower() + base[1:],
                NOT_adj = NOT_adj,
                adj     = adj,
            )

    if not_at:
        sentence = "NOT " + sentence

    return sentence


# --- two-step ------------------------------------------------------

def generate_base(adj_sentiment="positive", seed=SEED):
    """
    Step 1 — sample adjectives and sentiments. Returns a fixed DataFrame.
    The output of this function should be reused across experiments.
    """
    rng_sent = random.Random(seed)
    rng_adj  = random.Random(seed + 1)

    sent_list = [rng_sent.choice(["positive", "negative"]) if adj_sentiment == "both"
                 else adj_sentiment for _ in SENTENCE_BASES]

    records = []
    for i, ((domain, base), sent) in enumerate(zip(SENTENCE_BASES, sent_list)):
        adj = rng_adj.choice(ADJECTIVES[domain][sent])
        records.append({
            "id":           i + 1,
            "domain":       domain,
            "base":         base,
            "adj":          adj,
            "adj_sentiment": sent,
        })

    return pd.DataFrame(records)


def apply_not(base_df, n_not=1, contraction=False, not_at=False, aux_not=False,
              template_idx=None, seed=SEED):
    """
    Step 2 — apply NOT transformations to a fixed base DataFrame.
    Always produces the same adjectives; only the sentence form changes.
    """
    rng     = random.Random(seed)
    records = []

    for _, row in base_df.iterrows():
        total_not = n_not + (1 if not_at else 0)
        label     = row.adj_sentiment if total_not % 2 == 0 else (
                    "negative" if row.adj_sentiment == "positive" else "positive")
        sentence  = _build_sentence(row.base, row.adj, n_not, contraction,
                                    not_at, aux_not, rng)
        records.append({
            "id":            row.id,
            "domain":        row.domain,
            "base":          row.base,
            "adj":           row.adj,
            "adj_sentiment": row.adj_sentiment,
            "n_not":         n_not,
            "contraction":   contraction,
            "not_at":        not_at,
            "aux_not":       aux_not,
            "label":         label,
            "sentence":      sentence,
        })

    return pd.DataFrame(records)


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")