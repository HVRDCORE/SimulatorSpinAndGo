"""
Реализация состояния игры для симуляций покера в формате Spin and Go.
"""

from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any
import copy
import random
import json
import numpy as np

from pokersim.game.card import Card, Rank, Suit
from pokersim.game.deck import Deck
from pokersim.game.evaluator import HandEvaluator


class ActionType(Enum):
    """Перечисление возможных действий игрока."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()

    def __str__(self) -> str:
        return self.name.capitalize()

    def __repr__(self) -> str:
        return str(self)


class Action:
    """
    Действие, выполненное игроком в покерной игре.

    Атрибуты:
        action_type (ActionType): Тип действия.
        amount (int, optional): Сумма ставки или рейза, если применимо.
    """

    def __init__(self, action_type: ActionType, amount: int = 0):
        """
        Инициализация действия.

        Аргументы:
            action_type (ActionType): Тип действия.
            amount (int, optional): Сумма ставки или рейза, если применимо.
        """
        self.action_type = action_type
        self.amount = amount

    def __str__(self) -> str:
        """Возвращает строковое представление действия."""
        if self.action_type in {ActionType.BET, ActionType.RAISE}:
            return f"{self.action_type} {self.amount}"
        return str(self.action_type)

    def __repr__(self) -> str:
        """Возвращает строковое представление действия."""
        return str(self)

    def __eq__(self, other) -> bool:
        """Проверяет равенство двух действий."""
        if not isinstance(other, Action):
            return False
        return (self.action_type == other.action_type and
                self.amount == other.amount)


class Stage(Enum):
    """Перечисление стадий игры."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()

    def __str__(self) -> str:
        return self.name.capitalize()

    def __repr__(self) -> str:
        return str(self)


class GameState:
    """
    Состояние покерной игры в формате Spin and Go.

    Атрибуты:
        num_players (int): Количество игроков в игре (по умолчанию 3 для Spin and Go).
        small_blind (int): Размер малого блайнда.
        big_blind (int): Размер большого блайнда.
        stacks (List[int]): Стек фишек каждого игрока (фиксирован на 500).
        deck (Deck): Колода карт.
        hole_cards (List[List[Card]]): Приватные карты каждого игрока.
        community_cards (List[Card]): Общие карты.
        pot (int): Размер банка.
        current_bets (List[int]): Текущие ставки каждого игрока.
        stage (Stage): Текущая стадия игры.
        button (int): Игрок с баттоном дилера.
        current_player (int): Игрок, который должен действовать.
        last_raiser (int): Последний игрок, сделавший рейз.
        min_raise (int): Минимальная сумма рейза.
        active (List[bool]): Активность игроков (не сбросили карты).
        history (List[Tuple[int, Action]]): История действий.
        stage_history (List[List[Tuple[int, Action]]]): История действий по стадиям.
        payouts (List[int]): Выплаты каждому игроку в конце раздачи.
    """

    def __init__(self, num_players: int = 3, small_blind: int = 5, big_blind: int = 10,
                 initial_stacks: List[int] = None, button: int = 0):
        """
        Инициализация состояния игры.

        Аргументы:
            num_players (int, optional): Количество игроков. По умолчанию 3.
            small_blind (int, optional): Размер малого блайнда. По умолчанию 5.
            big_blind (int, optional): Размер большого блайнда. По умолчанию 10.
            initial_stacks (List[int], optional): Начальные стеки. Игнорируется, фиксировано 500.
            button (int, optional): Игрок с баттоном дилера. По умолчанию 0.
        """
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Фиксированные стеки для Spin and Go
        self.stacks = [500] * num_players

        self.deck = Deck()
        self.hole_cards = [[] for _ in range(num_players)]
        self.community_cards = []
        self.pot = 0
        self.current_bets = [0] * num_players
        self.stage = Stage.PREFLOP
        self.button = button
        self.active = [True] * num_players
        self.last_raiser = -1
        self.min_raise = big_blind
        self.history = []
        self.stage_history = [[] for _ in range(5)]  # По одной для каждой стадии + шоудаун
        self.payouts = [0] * num_players

        # Установка текущего игрока после инициализации active
        self.current_player = self._first_to_act()

        # Раздача приватных карт
        self._deal_hole_cards()

        # Постановка блайндов
        self._post_blinds()

    def _deal_hole_cards(self) -> None:
        """Раздача приватных карт каждому игроку."""
        for i in range(self.num_players):
            self.hole_cards[i] = self.deck.deal(2)

    def _post_blinds(self) -> None:
        """Постановка блайндов."""
        sb_pos = (self.button + 1) % self.num_players
        bb_pos = (self.button + 2) % self.num_players

        # Малый блайнд
        sb_amount = min(self.small_blind, self.stacks[sb_pos])
        self.stacks[sb_pos] -= sb_amount
        self.current_bets[sb_pos] = sb_amount
        self.pot += sb_amount
        self.history.append((sb_pos, Action(ActionType.BET, sb_amount)))
        self.stage_history[0].append((sb_pos, Action(ActionType.BET, sb_amount)))

        # Большой блайнд
        bb_amount = min(self.big_blind, self.stacks[bb_pos])
        self.stacks[bb_pos] -= bb_amount
        self.current_bets[bb_pos] = bb_amount
        self.pot += bb_amount
        self.history.append((bb_pos, Action(ActionType.BET, bb_amount)))
        self.stage_history[0].append((bb_pos, Action(ActionType.BET, bb_amount)))

        self.last_raiser = bb_pos

    def _first_to_act(self) -> int:
        """Определение первого игрока для действия."""
        pos = (self.button + 3) % self.num_players  # После блайндов на префлопе
        while not self.active[pos] or self.stacks[pos] == 0:
            pos = (pos + 1) % self.num_players
        return pos

    def _next_player(self) -> int:
        """Определение следующего игрока для действия."""
        for i in range(1, self.num_players):
            p = (self.current_player + i) % self.num_players
            if self.active[p] and self.stacks[p] > 0:
                return p
        return -1  # Нет активных игроков

    def _stage_complete(self) -> bool:
        """Проверка завершения текущей стадии."""
        if sum(self.active) <= 1:
            return True

        active_players = [i for i, active in enumerate(self.active) if active and self.stacks[i] > 0]
        if not active_players:
            return True

        max_bet = max(self.current_bets[i] for i in active_players)
        for p in active_players:
            if self.current_bets[p] < max_bet and self.stacks[p] > 0:
                return False

        # Проверка, что все игроки действовали после последнего рейза
        if self.last_raiser != -1:
            last_action_pos = self.history[-1][0] if self.history else -1
            if last_action_pos != self.last_raiser:
                return False

        return True

    def _advance_stage(self) -> None:
        """Переход к следующей стадии игры."""
        # Перемещение ставок в банк
        self.pot += sum(self.current_bets)
        self.current_bets = [0] * self.num_players

        # Сброс последнего рейзера и минимального рейза
        self.last_raiser = -1
        self.min_raise = self.big_blind

        # Раздача общих карт
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

        # Установка первого игрока для действия
        self.current_player = self._first_to_act()

    def _showdown(self) -> None:
        """Определение победителя на шоудауне."""
        if sum(self.active) <= 1:
            winner = self.active.index(True) if True in self.active else -1
            if winner != -1:
                self.payouts[winner] = self.pot
            return

        active_players = [i for i, active in enumerate(self.active) if active]
        if not active_players:
            return

        best_hands = []
        for p in active_players:
            hand = self.hole_cards[p] + self.community_cards
            if len(hand) >= 5:
                rank, desc = HandEvaluator.evaluate_hand(hand)
                best_hands.append((p, rank, hand))

        if not best_hands:
            return

        max_rank = max(hand[1] for hand in best_hands)
        winners = [hand[0] for hand in best_hands if hand[1] == max_rank]

        if len(winners) > 1:
            # Сравнение кикеров при равных рангах
            final_winners = []
            max_kicker_score = -1
            for p in winners:
                hand = self.hole_cards[p] + self.community_cards
                kicker_score = max(card.rank.value for card in hand)
                if kicker_score > max_kicker_score:
                    max_kicker_score = kicker_score
                    final_winners = [p]
                elif kicker_score == max_kicker_score:
                    final_winners.append(p)
            winners = final_winners

        pot_per_winner = self.pot // len(winners)
        for winner in winners:
            self.payouts[winner] = pot_per_winner

    def get_legal_actions(self) -> List[Action]:
        """
        Получение списка легальных действий для текущего игрока.

        Возвращает:
            List[Action]: Список легальных действий.
        """
        if (self.current_player == -1 or
            self.stage == Stage.SHOWDOWN or
            not self.active[self.current_player] or
            self.stacks[self.current_player] <= 0):
            return []

        actions = []
        max_bet = max(self.current_bets) if self.current_bets else 0
        current_bet = self.current_bets[self.current_player]
        stack = self.stacks[self.current_player]

        # Фолд всегда легален, если игрок активен
        actions.append(Action(ActionType.FOLD))

        # Чек легален, если ставка игрока равна максимальной
        if max_bet == current_bet:
            actions.append(Action(ActionType.CHECK))

        # Колл легален, если максимальная ставка выше
        if max_bet > current_bet:
            call_amount = min(max_bet - current_bet, stack)
            if call_amount > 0:
                actions.append(Action(ActionType.CALL, call_amount))

        # Бет легален, если никто не сделал ставку
        if max_bet == 0 and stack > 0:
            actions.append(Action(ActionType.BET, min(self.big_blind, stack)))
            if stack > self.big_blind:
                actions.append(Action(ActionType.BET, stack))

        # Рейз легален, если есть ставка и у игрока достаточно фишек
        if max_bet > 0 and stack > (max_bet - current_bet):
            min_raise_amount = min(self.min_raise + max_bet, stack)
            actions.append(Action(ActionType.RAISE, min_raise_amount))
            if stack > min_raise_amount:
                actions.append(Action(ActionType.RAISE, stack))

        return actions

    def apply_action(self, action: Action) -> 'GameState':
        """
        Применение действия к состоянию игры.

        Аргументы:
            action (Action): Действие для применения.

        Возвращает:
            GameState: Новое состояние игры.

        Вызывает:
            ValueError: Если действие нелегально.
        """
        if action not in self.get_legal_actions():
            raise ValueError(f"Нелегальное действие: {action}")

        new_state = copy.deepcopy(self)
        player = new_state.current_player

        if action.action_type == ActionType.FOLD:
            new_state.active[player] = False

        elif action.action_type == ActionType.CHECK:
            pass

        elif action.action_type == ActionType.CALL:
            call_amount = min(max(new_state.current_bets) - new_state.current_bets[player],
                              new_state.stacks[player])
            new_state.stacks[player] -= call_amount
            new_state.current_bets[player] += call_amount
            new_state.pot += call_amount

        elif action.action_type == ActionType.BET:
            new_state.stacks[player] -= action.amount
            new_state.current_bets[player] = action.amount
            new_state.pot += action.amount
            new_state.last_raiser = player
            new_state.min_raise = action.amount

        elif action.action_type == ActionType.RAISE:
            additional_amount = action.amount - new_state.current_bets[player]
            new_state.stacks[player] -= additional_amount
            new_state.current_bets[player] = action.amount
            new_state.pot += additional_amount
            new_state.last_raiser = player
            new_state.min_raise = action.amount - max(new_state.current_bets)

        # Запись действия в историю
        new_state.history.append((player, action))
        new_state.stage_history[new_state.stage.value - 1].append((player, action))

        # Обновление текущего игрока
        new_state.current_player = new_state._next_player()

        # Проверка завершения стадии
        if new_state._stage_complete():
            new_state._advance_stage()

        return new_state

    def is_terminal(self) -> bool:
        """
        Проверка, является ли состояние игры терминальным.

        Возвращает:
            bool: True, если состояние терминально, иначе False.
        """
        active_count = sum(1 for i, active in enumerate(self.active) if active and self.stacks[i] > 0)
        return self.stage == Stage.SHOWDOWN or active_count <= 1

    def get_observation(self, player_id: int) -> Dict[str, Any]:
        """
        Получение наблюдения состояния игры с точки зрения игрока.

        Аргументы:
            player_id (int): ID игрока.

        Возвращает:
            Dict[str, Any]: Наблюдение.
        """
        observation = {
            'player_id': player_id,
            'num_players': self.num_players,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'stacks': self.stacks,
            'hole_cards': self.hole_cards[player_id] if 0 <= player_id < self.num_players else [],
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
        """
        Получение выплат для каждого игрока.

        Возвращает:
            List[int]: Выплаты.
        """
        return self.payouts

    def get_rewards(self) -> List[float]:
        """
        Получение наград для каждого игрока.

        Возвращает:
            List[float]: Награды.
        """
        if not self.is_terminal():
            return [0.0] * self.num_players

        rewards = [0.0] * self.num_players
        for i, payout in enumerate(self.payouts):
            rewards[i] = float(payout - self.current_bets[i])
        return rewards

    def to_feature_vector(self, player_id: int) -> np.ndarray:
        """
        Преобразование состояния игры в вектор признаков с точки зрения игрока.

        Аргументы:
            player_id (int): ID игрока.

        Возвращает:
            np.ndarray: Вектор признаков.
        """
        # Признаки:
        # - One-hot кодирование приватных карт (2 карты x 52 варианта)
        # - One-hot кодирование общих карт (5 карт x 52 варианта)
        # - Размер банка (нормализован)
        # - Размеры стеков (нормализованы для каждого игрока)
        # - Текущие ставки (нормализованы для каждого игрока)
        # - Стадия игры (one-hot кодирование)
        # - Позиция баттона (one-hot кодирование)
        # - Текущий игрок (one-hot кодирование)
        # - Активные игроки (бинарно для каждого игрока)

        total_chips = max(sum(self.stacks) + self.pot, 1)

        # One-hot кодирование приватных карт
        hole_cards_features = np.zeros(2 * 52)
        if 0 <= player_id < self.num_players:
            for i, card in enumerate(self.hole_cards[player_id]):
                hole_cards_features[i * 52 + card.to_int()] = 1

        # One-hot кодирование общих карт
        community_cards_features = np.zeros(5 * 52)
        for i, card in enumerate(self.community_cards):
            community_cards_features[i * 52 + card.to_int()] = 1

        # Размер банка (нормализован)
        pot_feature = np.array([self.pot / total_chips])

        # Размеры стеков (нормализованы для каждого игрока)
        stack_features = np.array([stack / total_chips for stack in self.stacks])

        # Текущие ставки (нормализованы для каждого игрока)
        bet_features = np.array([bet / total_chips for bet in self.current_bets])

        # Стадия игры (one-hot кодирование)
        stage_features = np.zeros(5)
        stage_features[self.stage.value - 1] = 1

        # Позиция баттона (one-hot кодирование)
        button_features = np.zeros(self.num_players)
        button_features[self.button] = 1

        # Текущий игрок (one-hot кодирование)
        current_player_features = np.zeros(self.num_players)
        if self.current_player != -1:
            current_player_features[self.current_player] = 1

        # Активные игроки (бинарно для каждого игрока)
        active_features = np.array([1 if active else 0 for active in self.active])

        # Объединение всех признаков
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

    def __str__(self) -> str:
        """Возвращает строковое представление состояния игры."""
        s = f"Стадия: {self.stage}\n"
        s += f"Банк: {self.pot}\n"
        s += f"Общие карты: {self.community_cards}\n"
        s += "Игроки:\n"

        for i in range(self.num_players):
            s += f"  Игрок {i}: "
            if i == self.button:
                s += "(Баттон) "
            if i == self.current_player:
                s += "(Ходит) "
            s += f"Стек: {self.stacks[i]} "
            s += f"Ставка: {self.current_bets[i]} "
            if self.active[i]:
                s += f"Карты: {self.hole_cards[i]}\n"
            else:
                s += "Сбросил\n"

        return s