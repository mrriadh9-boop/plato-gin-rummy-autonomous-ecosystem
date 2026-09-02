"""
Deterministic Disjoint Meld Solver and Deadwood Evaluator for Gin Rummy.
"""
from __future__ import annotations

import itertools
from typing import List, Tuple, Set, Dict, Optional, Any

from vision import RANKS, SUITS

RANK_TO_INT: Dict[str, int] = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13,
}

INT_TO_RANK: Dict[int, str] = {
    1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
    8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K",
}


def parse_card(card_str: str) -> Tuple[int, str]:
    """Parses card string (e.g. '9♦', '10♥', 'KH') into (rank_int, suit_str)."""
    card_str = card_str.strip()
    suit = card_str[-1]
    rank_part = card_str[:-1]
    rank_int = RANK_TO_INT.get(rank_part.upper(), 0)
    return rank_int, suit


def card_to_string(rank_int: int, suit: str) -> str:
    """Formats (rank_int, suit) to string (e.g. (9, '♦') -> '9♦')."""
    return f"{INT_TO_RANK.get(rank_int, str(rank_int))}{suit}"


def card_deadwood_value(card_str: str) -> int:
    """Returns deadwood point value of a single card (A=1, 2..10=face value, J/Q/K=10)."""
    rank_int, _ = parse_card(card_str)
    if rank_int <= 10:
        return max(1, rank_int)
    return 10


class MeldSolver:
    """Combinatorial solver for disjoint melds and deadwood calculation."""

    def find_all_melds(self, cards: List[str]) -> List[List[str]]:
        """Finds all valid sets and runs formable from the input card list."""
        melds: List[List[str]] = []

        # 1. Sets: 3 or 4 of the same rank
        rank_buckets: Dict[int, List[str]] = {}
        for c in cards:
            r, s = parse_card(c)
            rank_buckets.setdefault(r, []).append(c)

        for r, bucket in rank_buckets.items():
            if len(bucket) >= 3:
                for comb in itertools.combinations(bucket, 3):
                    melds.append(list(comb))
                if len(bucket) == 4:
                    melds.append(list(bucket))

        # 2. Runs: 3 or more consecutive ranks of same suit (Ace low: A-2-3)
        suit_buckets: Dict[str, List[Tuple[int, str]]] = {}
        for c in cards:
            r, s = parse_card(c)
            suit_buckets.setdefault(s, []).append((r, c))

        for s, bucket in suit_buckets.items():
            bucket.sort(key=lambda x: x[0])
            n = len(bucket)
            if n >= 3:
                for start_idx in range(n):
                    for end_idx in range(start_idx + 3, n + 1):
                        sub = bucket[start_idx:end_idx]
                        is_run = all(sub[i][0] == sub[0][0] + i for i in range(len(sub)))
                        if is_run:
                            melds.append([item[1] for item in sub])

        return melds

    def calculate_deadwood(self, cards: List[str]) -> Tuple[int, List[List[str]], List[str]]:
        """
        Computes minimum deadwood points and optimal disjoint melds partition.
        Returns: (min_deadwood_points, best_melds, deadwood_cards)
        """
        all_melds = self.find_all_melds(cards)
        total_card_points = sum(card_deadwood_value(c) for c in cards)

        if not all_melds:
            return total_card_points, [], list(cards)

        best_deadwood = total_card_points
        best_melds: List[List[str]] = []

        def search(meld_idx: int, used_cards: Set[str], current_melds: List[List[str]]):
            nonlocal best_deadwood, best_melds

            current_deadwood = sum(card_deadwood_value(c) for c in cards if c not in used_cards)
            if current_deadwood < best_deadwood:
                best_deadwood = current_deadwood
                best_melds = list(current_melds)

            for i in range(meld_idx, len(all_melds)):
                meld = all_melds[i]
                if not any(c in used_cards for c in meld):
                    new_used = used_cards.union(meld)
                    search(i + 1, new_used, current_melds + [meld])

        search(0, set(), [])

        melded_set = set(c for meld in best_melds for c in meld)
        deadwood_cards = [c for c in cards if c not in melded_set]
        return best_deadwood, best_melds, deadwood_cards

    def get_layoff_cards(self, melds: List[List[str]]) -> Set[str]:
        """Computes all standard cards that can be laid off onto existing melds."""
        layoffs: Set[str] = set()
        for meld in melds:
            parsed = [parse_card(c) for c in meld]
            ranks = [p[0] for p in parsed]
            suits = [p[1] for p in parsed]

            if len(set(ranks)) == 1:
                # Set: any remaining suit of same rank
                r = ranks[0]
                present = set(suits)
                for s in SUITS:
                    if s not in present:
                        layoffs.add(card_to_string(r, s))
            elif len(set(suits)) == 1:
                # Run: extending lower or upper rank
                s = suits[0]
                min_r = min(ranks)
                max_r = max(ranks)
                if min_r > 1:
                    layoffs.add(card_to_string(min_r - 1, s))
                if max_r < 13:
                    layoffs.add(card_to_string(max_r + 1, s))
        return layoffs
