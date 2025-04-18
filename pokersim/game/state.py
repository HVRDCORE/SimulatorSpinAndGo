from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any
import copy
import numpy as np
import logging

from pokersim.game.card import Card, Rank, Suit
from pokersim.game.deck import Deck
from pokersim.game.evaluator import HandEvaluator
from pokersim.config.config_manager import get_config

logger = logging.getLogger(__name__)

class ActionType(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()

    def __str__(self) -> str:
        return self.name.capitalize()

class Action:
    def __init__(self, action_type: ActionType, amount: int = 0):
        self.action_type = action_type
        self.amount = amount

    def __str__(self) -> str:
        if self.action_type in {ActionType.BET, ActionType.RAISE}:
            return f"{self.action_type} {self.amount}"
        return str(self.action_type)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            return False
        return self.action_type == other.action_type and self.amount == other.amount

class Stage(Enum):
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()

    def __str__(self) -> str:
        return self.name.capitalize()

class GameState:
    def __init__(
        self,
        num_players: int,
        small_blind: int,
        big_blind: int,
        initial_stacks: List[int],
        button: int,
        player_ids: List[int]
    ):
        config = get_config()
        self.num_players = num_players
        self.small_blind = config.get('game.stakes.small_blind', small_blind)
        self.big_blind = config.get('game.stakes.big_blind', big_blind)
        if self.num_players < 2:
            raise ValueError("Количество игроков должно быть не менее 2")
        if len(initial_stacks) != self.num_players or len(player_ids) != self.num_players:
            raise ValueError("Длина initial_stacks и player_ids должна соответствовать num_players")

        self.stacks = initial_stacks.copy()
        self.player_ids = player_ids.copy()
        self.deck = Deck()
        self.hole_cards = [[] for _ in range(self.num_players)]
        self.community_cards = []
        self.pot = 0
        self.current_bets = [0] * self.num_players
        self.stage = Stage.PREFLOP
        self.button = button % self.num_players
        self.active = [True] * self.num_players
        self.last_raiser = -1
        self.min_raise = self.big_blind
        self.history = []
        self.stage_history = [[] for _ in range(5)]
        self.payouts = [0] * self.num_players

        self.current_player = self._first_to_act()
        self._deal_hole_cards()
        self._post_blinds()
        logger.debug(f"Инициализировано GameState с {self.num_players} игроками")

    def _deal_hole_cards(self) -> None:
        for i in range(self.num_players):
            if self.stacks[i] > 0:
                self.hole_cards[i] = self.deck.deal(2)

    def _post_blinds(self) -> None:
        sb_pos = (self.button + 1) % self.num_players
        bb_pos = (self.button + 2) % self.num_players

        sb_amount = min(self.small_blind, self.stacks[sb_pos])
        self.stacks[sb_pos] -= sb_amount
        self.current_bets[sb_pos] = sb_amount
        self.pot += sb_amount
        self.history.append((self.player_ids[sb_pos], Action(ActionType.BET, sb_amount)))
        self.stage_history[0].append((self.player_ids[sb_pos], Action(ActionType.BET, sb_amount)))

        bb_amount = min(self.big_blind, self.stacks[bb_pos])
        self.stacks[bb_pos] -= bb_amount
        self.current_bets[bb_pos] = bb_amount
        self.pot += bb_amount
        self.history.append((self.player_ids[bb_pos], Action(ActionType.BET, bb_amount)))
        self.stage_history[0].append((self.player_ids[bb_pos], Action(ActionType.BET, bb_amount)))

        self.last_raiser = self.player_ids[bb_pos]
        logger.debug(f"Блайнды размещены: SB={sb_amount}, BB={bb_amount}")

    def _first_to_act(self) -> int:
        pos = (self.button + 3) % self.num_players
        while not self.active[pos] or self.stacks[pos] == 0:
            pos = (pos + 1) % self.num_players
        return self.player_ids[pos]

    def _next_player(self) -> int:
        current_idx = self.player_ids.index(self.current_player)
        for i in range(1, self.num_players):
            pos = (current_idx + i) % self.num_players
            if self.active[pos] and self.stacks[pos] > 0:
                return self.player_ids[pos]
        return -1

    def _stage_complete(self) -> bool:
        if sum(self.active) <= 1:
            logger.debug("Стадия завершена: остался один активный игрок")
            return True

        active_players = [i for i, active in enumerate(self.active) if active and self.stacks[i] > 0]
        if not active_players:
            logger.debug("Стадия завершена: нет активных игроков")
            return True

        max_bet = max(self.current_bets[i] for i in active_players)
        for p in active_players:
            if self.current_bets[p] < max_bet and self.stacks[p] > 0:
                return False

        all_in_count = sum(1 for p in active_players if self.stacks[p] == 0 and self.active[p])
        if all_in_count == len(active_players):
            logger.debug("Стадия завершена: все активные игроки в олл-ин")
            return True

        if self.last_raiser != -1:
            last_action_id = self.history[-1][0] if self.history else -1
            if last_action_id != self.last_raiser:
                return False

        logger.debug("Стадия завершена: все ставки выровнены")
        return True

    def _advance_stage(self) -> None:
        self.pot += sum(self.current_bets)
        self.current_bets = [0] * self.num_players
        self.last_raiser = -1
        self.min_raise = self.big_blind

        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self.community_cards.extend(self.deck.deal(3))
        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self.community_cards.extend(self.deck.deal(1))
        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self.community_cards.extend(self.deck.deal(1))
        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN
            self._showdown()
        logger.debug(f"Переход к стадии: {self.stage}")

        self.current_player = self._first_to_act()

    def _showdown(self) -> None:
        active_players = [i for i, active in enumerate(self.active) if active and self.stacks[i] > 0]
        if not active_players:
            logger.debug("Вскрытие: нет активных игроков, банк не распределён")
            return

        if sum(self.active) <= 1:
            winner_idx = self.active.index(True) if True in self.active else -1
            if winner_idx != -1:
                self.payouts[winner_idx] = self.pot
                logger.info(f"Вскрытие: единственный активный игрок {self.player_ids[winner_idx]}, банк={self.pot}")
            else:
                logger.debug("Вскрытие: нет активных игроков")
            return

        best_hands = []
        for p in active_players:
            hand = self.hole_cards[p] + self.community_cards
            if len(hand) >= 5:
                rank, best_hand = HandEvaluator.evaluate_hand(hand)
                best_hands.append((p, rank, best_hand, hand))
            else:
                logger.warning(f"Недостаточно карт для игрока {p}: {hand}")
                continue

        if not best_hands:
            logger.debug("Вскрытие: нет валидных рук, делим банк между активными игроками")
            pot_per_player = self.pot // len(active_players)
            for p in active_players:
                self.payouts[p] = pot_per_player
            logger.info(f"Вскрытие: банк {self.pot} разделён между {active_players}, каждому по {pot_per_player}")
            return

        max_rank = max(hand[1] for hand in best_hands)
        candidates = [(p, hand) for p, rank, _, hand in best_hands if rank == max_rank]

        winners = []
        if len(candidates) == 1:
            winners = [candidates[0][0]]
        else:
            best_score = -1
            for p, hand in candidates:
                score = sum(card.rank.value for card in hand[-5:])
                if score > best_score:
                    winners = [p]
                    best_score = score
                elif score == best_score:
                    winners.append(p)

        pot_per_winner = self.pot // len(winners)
        for winner in winners:
            self.payouts[winner] = pot_per_winner
        logger.info(f"Вскрытие: победители {winners}, банк на победителя={pot_per_winner}, общий банк={self.pot}")
        # Добавляем проверку общей суммы выплат
        total_payouts = sum(self.payouts)
        if total_payouts > self.pot:
            logger.error(f"Ошибка: сумма выплат ({total_payouts}) превышает банк ({self.pot})")
            # Корректируем выплаты пропорционально
            scale = self.pot / total_payouts
            for i in range(self.num_players):
                self.payouts[i] = int(self.payouts[i] * scale)
            logger.info(f"Выплаты скорректированы: {self.payouts}")

    def get_legal_actions(self) -> List[Action]:
        if (self.current_player == -1 or
            self.stage == Stage.SHOWDOWN or
            self.current_player not in self.player_ids or
            not self.active[self.player_ids.index(self.current_player)] or
            self.stacks[self.player_ids.index(self.current_player)] <= 0):
            return []

        player_idx = self.player_ids.index(self.current_player)
        actions = []
        max_bet = max(self.current_bets) if self.current_bets else 0
        current_bet = self.current_bets[player_idx]
        stack = self.stacks[player_idx]

        actions.append(Action(ActionType.FOLD))
        if max_bet == current_bet:
            actions.append(Action(ActionType.CHECK))
        if max_bet > current_bet:
            call_amount = min(max_bet - current_bet, stack)
            if call_amount > 0:
                actions.append(Action(ActionType.CALL, call_amount))
        if max_bet == 0 and stack > 0:
            min_bet = min(self.big_blind, stack)
            actions.append(Action(ActionType.BET, min_bet))
            half_stack = max(min_bet, stack // 2)
            if half_stack > min_bet:
                actions.append(Action(ActionType.BET, half_stack))
        if max_bet > 0 and stack > (max_bet - current_bet):
            min_raise_amount = min(max_bet + self.min_raise, stack)
            if min_raise_amount > max_bet:
                actions.append(Action(ActionType.RAISE, min_raise_amount))
            double_min_raise = min(2 * self.min_raise + max_bet, stack)
            if double_min_raise > min_raise_amount:
                actions.append(Action(ActionType.RAISE, double_min_raise))

        return actions

    def apply_action(self, action: Action) -> 'GameState':
        if action not in self.get_legal_actions():
            raise ValueError(f"Недопустимое действие: {action}")

        new_state = copy.deepcopy(self)
        player_idx = new_state.player_ids.index(new_state.current_player)

        logger.debug(f"Игрок {new_state.current_player} выполняет действие: {action}")

        if action.action_type == ActionType.FOLD:
            new_state.active[player_idx] = False
        elif action.action_type == ActionType.CHECK:
            pass
        elif action.action_type == ActionType.CALL:
            call_amount = min(max(new_state.current_bets) - new_state.current_bets[player_idx],
                              new_state.stacks[player_idx])
            new_state.stacks[player_idx] -= call_amount
            new_state.current_bets[player_idx] += call_amount
            new_state.pot += call_amount
        elif action.action_type == ActionType.BET:
            if action.amount > new_state.stacks[player_idx]:
                raise ValueError(f"Ставка {action.amount} превышает стек игрока {new_state.stacks[player_idx]}")
            new_state.stacks[player_idx] -= action.amount
            new_state.current_bets[player_idx] = action.amount
            new_state.pot += action.amount
            new_state.last_raiser = new_state.current_player
            new_state.min_raise = action.amount
        elif action.action_type == ActionType.RAISE:
            additional_amount = action.amount - new_state.current_bets[player_idx]
            if additional_amount > new_state.stacks[player_idx]:
                raise ValueError(f"Рейз {additional_amount} превышает стек игрока {new_state.stacks[player_idx]}")
            new_state.stacks[player_idx] -= additional_amount
            new_state.current_bets[player_idx] = action.amount
            new_state.pot += additional_amount
            new_state.last_raiser = new_state.current_player
            new_state.min_raise = action.amount - max(new_state.current_bets)

        new_state.history.append((new_state.current_player, action))
        new_state.stage_history[new_state.stage.value - 1].append((new_state.current_player, action))
        new_state.current_player = new_state._next_player()

        if new_state._stage_complete():
            new_state._advance_stage()

        # Проверка инварианта: сумма стеков + банк = 1500
        total_chips = sum(new_state.stacks) + new_state.pot
        if total_chips != 1500:
            logger.error(
                f"Нарушение инварианта: сумма стеков ({sum(new_state.stacks)}) + банк ({new_state.pot}) = {total_chips}, ожидается 1500")
            # Можно добавить корректировку, но лучше найти источник ошибки

        return new_state

    def is_terminal(self) -> bool:
        active_count = sum(1 for i in range(self.num_players) if self.active[i] and self.stacks[i] > 0)
        is_terminal = self.stage == Stage.SHOWDOWN or active_count <= 1
        if is_terminal:
            logger.debug(f"Игра завершена: стадия={self.stage}, активных игроков={active_count}")
        return is_terminal

    def get_observation(self, player_id: int) -> Dict[str, Any]:
        player_idx = self.player_ids.index(player_id) if player_id in self.player_ids else -1
        observation = {
            'player_id': player_id,
            'num_players': self.num_players,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'stacks': self.stacks,
            'hole_cards': self.hole_cards[player_idx] if player_idx != -1 else [],
            'community_cards': self.community_cards,
            'pot': self.pot,
            'current_bets': self.current_bets,
            'stage': self.stage,
            'button': self.button,
            'current_player': self.current_player,
            'active': self.active,
            'history': self.history,
            'legal_actions': self.get_legal_actions() if self.current_player == player_id else []
        }
        return observation

    def get_payouts(self) -> List[int]:
        logger.debug(f"Возвращение выплат: {self.payouts}, текущий банк: {self.pot}, стеки: {self.stacks}")
        total_payouts = sum(self.payouts)
        if total_payouts > self.pot:
            logger.error(f"Ошибка: сумма выплат ({total_payouts}) превышает банк ({self.pot})")
            # Возвращаем пропорциональные выплаты
            scale = self.pot / total_payouts if total_payouts > 0 else 0
            adjusted_payouts = [int(p * scale) for p in self.payouts]
            logger.info(f"Выплаты скорректированы: {adjusted_payouts}")
            return adjusted_payouts
        return self.payouts

    def get_rewards(self) -> List[float]:
        if not self.is_terminal():
            return [0.0] * self.num_players

        rewards = [0.0] * self.num_players
        for i in range(self.num_players):
            rewards[i] = float(self.payouts[i] - self.current_bets[i])
        logger.debug(f"Возвращение наград: {rewards}")
        return rewards

    def to_feature_vector(self, player_id: int) -> np.ndarray:
        total_chips = max(sum(self.stacks) + self.pot, 1)
        hole_cards_features = np.zeros(2 * 52)
        player_idx = self.player_ids.index(player_id) if player_id in self.player_ids else -1
        if player_idx != -1:
            for i, card in enumerate(self.hole_cards[player_idx]):
                hole_cards_features[i * 52 + card.to_int()] = 1
        community_cards_features = np.zeros(5 * 52)
        for i, card in enumerate(self.community_cards):
            community_cards_features[i * 52 + card.to_int()] = 1
        pot_feature = np.array([self.pot / total_chips])
        stack_features = np.array([stack / total_chips for stack in self.stacks])
        bet_features = np.array([bet / total_chips for bet in self.current_bets])
        stage_features = np.zeros(5)
        stage_features[self.stage.value - 1] = 1
        button_features = np.zeros(self.num_players)
        button_features[self.button] = 1
        current_player_features = np.zeros(self.num_players)
        if self.current_player != -1:
            current_player_idx = self.player_ids.index(self.current_player)
            current_player_features[current_player_idx] = 1
        active_features = np.array([1 if active else 0 for active in self.active])
        features = np.concatenate([
            hole_cards_features,
            community_cards_features,
            pot_feature,
            stack_features,
            bet_features,
            stage_features,
            button_features,
            current_player_features,
            active_features
        ])
        return features

    def get_info_set_dimension(self) -> int:
        return (2 * 52 + 5 * 52 + 1 + self.num_players * 4 + 5)

    def get_info_set_vector(self) -> np.ndarray:
        if self.current_player == -1:
            return np.zeros(self.get_info_set_dimension())
        return self.to_feature_vector(self.current_player)

    def get_info_set_key(self) -> str:
        actions_str = "_".join(f"{pid}:{action}" for pid, action in self.history)
        return f"p{self.current_player}_s{self.stage.value}_{actions_str}"

    def is_chance_node(self) -> bool:
        return (self.stage in [Stage.PREFLOP, Stage.FLOP, Stage.TURN] and
                len(self.community_cards) < {Stage.PREFLOP: 0, Stage.FLOP: 3, Stage.TURN: 4}.get(self.stage, 0))

    def get_chance_outcomes(self) -> List[Tuple[Card, float]]:
        if not self.is_chance_node():
            return []

        used_cards = set()
        for player_cards in self.hole_cards:
            used_cards.update(card.to_int() for card in player_cards)
        used_cards.update(card.to_int() for card in self.community_cards)
        remaining_cards = [Card.from_int(i) for i in range(52) if i not in used_cards]

        if not remaining_cards:
            return []

        prob = 1.0 / len(remaining_cards)
        return [(card, prob) for card in remaining_cards]

    def get_utility(self) -> float:
        if not self.is_terminal() or self.current_player == -1:
            return 0.0
        player_idx = self.player_ids.index(self.current_player)
        return float(self.payouts[player_idx] - self.current_bets[player_idx])

    def get_expected_value(self, player: int) -> float:
        if player not in self.player_ids:
            return 0.0
        player_idx = self.player_ids.index(player)
        if self.is_terminal():
            return float(self.payouts[player_idx] - self.current_bets[player_idx])
        return float(self.stacks[player_idx] - self.current_bets[player_idx])

    def get_current_player(self) -> int:
        return self.current_player

    def __str__(self) -> str:
        s = f"Стадия: {self.stage}\n"
        s += f"Банк: {self.pot}\n"
        s += f"Общие карты: {self.community_cards}\n"
        s += "Игроки:\n"
        for i in range(self.num_players):
            player_id = self.player_ids[i]
            s += f"  Игрок {player_id}: "
            if i == self.button:
                s += "(Кнопка) "
            if player_id == self.current_player:
                s += "(Действует) "
            s += f"Стек: {self.stacks[i]} "
            s += f"Ставка: {self.current_bets[i]} "
            if self.active[i]:
                s += f"Карты: {self.hole_cards[i]}\n"
            else:
                s += "Сбросил\n"
        return s