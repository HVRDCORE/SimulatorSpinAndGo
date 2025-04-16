"""
Пример скрипта для обучения агента с использованием PPO в формате Spin and Go.
"""
import sys
import os
import random
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent
from pokersim.ml.models import PokerActorCritic
from pokersim.algorithms.ppo import PPOSolver
from pokersim.ml.advanced_agents import PPOAgent

def play_episode(agent, opponents, num_players: int = 3, verbose: bool = False) -> Dict:
    game_state = GameState(num_players=num_players, small_blind=5, big_blind=10)
    player_positions = list(range(num_players))
    random.shuffle(player_positions)

    agent_position = player_positions[0]
    opponent_positions = player_positions[1:]

    agent.player_id = agent_position
    for i, opp in enumerate(opponents):
        opp.player_id = opponent_positions[i]

    max_steps = 300  # Уменьшено для Spin and Go
    step = 0

    while not game_state.is_terminal():
        if step >= max_steps:
            print(f"Предупреждение: Достигнуто максимальное количество шагов ({max_steps}). Завершение эпизода.")
            break

        # Проверка активных игроков
        active_count = sum(1 for i, active in enumerate(game_state.active) if active and game_state.stacks[i] > 0)
        if active_count < 2 and not game_state.is_terminal():
            print(f"Предупреждение: Меньше 2 активных игроков ({active_count}). Завершение эпизода.")
            break

        current_player = game_state.current_player
        if current_player < 0 or current_player >= num_players:
            print(f"Ошибка: Неверный current_player {current_player}. Завершение эпизода.")
            break

        print(f"Шаг {step}: Текущий игрок: {current_player}, Стадия: {game_state.stage}, "
              f"Активные игроки: {game_state.active}, Ставки: {game_state.current_bets}, Банк: {game_state.pot}")
        legal_actions = game_state.get_legal_actions()
        print(f"Легальные действия: {legal_actions}")

        if not legal_actions and not game_state.is_terminal():
            print("Предупреждение: Нет легальных действий в нетерминальном состоянии. Завершение эпизода.")
            break

        action = None
        if current_player == agent_position:
            print("Агент действует...")
            action = agent.act(game_state)
            agent.observe(game_state)
            print(f"Агент выбрал действие: {action}")
        else:
            for opp in opponents:
                if opp.player_id == current_player:
                    action = opp.act(game_state)
                    print(f"Противник {opp.player_id} выбрал действие: {action}")
                    break
            else:
                temp_agent = RandomAgent(current_player)
                print(f"Создание временного RandomAgent для игрока {current_player}")
                action = temp_agent.act(game_state)
                print(f"Временный RandomAgent выбрал действие: {action}")

        if action is None:
            print("Ошибка: Действие не выбрано. Завершение эпизода.")
            break

        old_state = game_state
        try:
            game_state = game_state.apply_action(action)
        except ValueError as e:
            print(f"Ошибка применения действия {action}: {e}. Завершение эпизода.")
            break
        print(f"Применено действие: {action}")
        print(f"Состояние изменилось: {old_state.current_player} -> {game_state.current_player}, "
              f"Ставки: {old_state.current_bets} -> {game_state.current_bets}, "
              f"Стадия: {old_state.stage} -> {game_state.stage}, Банк: {game_state.pot}")

        step += 1

    agent.observe(game_state)
    reward = game_state.get_rewards()[agent_position]

    if verbose:
        print(f"Эпизод завершен: Награда агента: {reward}")

    return {
        'reward': reward,
        'terminal_state': game_state
    }

def evaluate_agent(agent, opponents, num_episodes: int = 100, num_players: int = 3, verbose: bool = False) -> Dict:
    total_reward = 0
    wins = 0

    for episode in range(num_episodes):
        if verbose and episode % 10 == 0:
            print(f"Оценка эпизода {episode}/{num_episodes}...")

        result = play_episode(agent, opponents, num_players, verbose=False)

        total_reward += result['reward']
        if result['reward'] > 0:
            wins += 1

    avg_reward = total_reward / num_episodes
    win_rate = wins / num_episodes

    if verbose:
        print(f"Оценка завершена: Процент побед: {win_rate:.4f}, Средняя награда: {avg_reward:.4f}")

    return {
        'avg_reward': avg_reward,
        'win_rate': win_rate,
        'total_reward': total_reward,
        'episodes_played': num_episodes
    }

def train_ppo(args):
    print("Начало обучения PPO для Spin and Go...")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Используемое устройство: {device}")

    agent = PPOAgent(
        player_id=0,
        game_state_class=GameState,
        num_players=args.num_players,
        device=device,
        epsilon=args.epsilon,
        framework="auto"
    )

    opponents = [
        RandomAgent(1),
        CallAgent(2),
    ]

    opponents = opponents[:args.num_players - 1]

    eval_opponents = [
        RuleBased2Agent(1, aggression=0.6, bluff_frequency=0.1),
        RuleBased1Agent(2, aggression=0.4)
    ]

    eval_opponents = eval_opponents[:args.num_players - 1]

    best_win_rate = 0.0
    training_metrics = {
        'episode_rewards': [],
        'win_rates': [],
        'avg_rewards': []
    }

    print(f"Начало обучения на {args.episodes} эпизодов...")

    for episode in range(args.episodes):
        if episode % 10 == 0:
            print(f"\nЭпизод {episode + 1}/{args.episodes}")

        result = play_episode(agent, opponents, num_players=args.num_players, verbose=False)

        training_metrics['episode_rewards'].append(result['reward'])

        if (episode + 1) % 10 == 0:
            avg_reward = sum(training_metrics['episode_rewards'][-10:]) / 10
            print(f"Средняя награда за последние 10 эпизодов: {avg_reward:.4f}")

        if (episode + 1) % args.eval_interval == 0:
            print("Оценка агента...")
            eval_metrics = evaluate_agent(agent, eval_opponents, num_episodes=args.eval_episodes,
                                         num_players=args.num_players, verbose=True)

            training_metrics['win_rates'].append(eval_metrics['win_rate'])
            training_metrics['avg_rewards'].append(eval_metrics['avg_reward'])

            print(f"Результаты оценки - Процент побед: {eval_metrics['win_rate']:.4f}, "
                  f"Средняя награда: {eval_metrics['avg_reward']:.4f}")

            if eval_metrics['win_rate'] > best_win_rate:
                best_win_rate = eval_metrics['win_rate']
                if args.save_model:
                    save_path = args.model_path
                    print(f"Сохранение лучшей модели в {save_path}")
                    agent.save(save_path)

        if episode > args.episodes // 2:
            agent.epsilon = max(0.01, agent.epsilon * 0.995)

    print("\nОбучение завершено!")

    print("\nПроведение финальной оценки...")
    final_metrics = evaluate_agent(agent, eval_opponents, num_episodes=args.eval_episodes*2,
                                  num_players=args.num_players, verbose=True)

    print("\nРезультаты финальной оценки:")
    print(f"Процент побед: {final_metrics['win_rate']:.4f}")
    print(f"Средняя награда: {final_metrics['avg_reward']:.4f}")
    print(f"Всего сыграно эпизодов: {final_metrics['episodes_played']}")

    return agent, training_metrics, final_metrics

def main():
    parser = argparse.ArgumentParser(description="Обучение покерного агента с использованием PPO для Spin and Go")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Количество эпизодов обучения")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="Скорость обучения")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Начальный уровень исследования")
    parser.add_argument("--num-players", type=int, default=3,
                        help="Количество игроков в игре (3 для Spin and Go)")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Оценка агента каждые N эпизодов")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Количество эпизодов для оценки")
    parser.add_argument("--save-model", action="store_true",
                        help="Сохранение лучшей модели во время обучения")
    parser.add_argument("--model-path", type=str, default="models/ppo_agent.pt",
                        help="Путь для сохранения модели")
    parser.add_argument("--cpu", action="store_true",
                        help="Принудительное использование CPU, даже если доступен CUDA")

    args = parser.parse_args()

    if args.save_model and args.model_path:
        model_dir = os.path.dirname(args.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        else:
            print("Предупреждение: model_path не содержит директории, используется текущая папка")
            args.model_path = os.path.join(".", os.path.basename(args.model_path) or "ppo_agent.pt")

    agent, training_metrics, final_metrics = train_ppo(args)

    print("\nОбучение PPO завершено!")
    print("Для дополнительных примеров и деталей см. README.md и директорию docs/.")

if __name__ == "__main__":
    main()