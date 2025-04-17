"""
Main entry point for poker AI training and simulation with configurable opponents,
optimized logging, and metrics visualization.
"""

import torch
import logging
import matplotlib.pyplot as plt
from typing import List, Optional
from pokersim.game.spingo import SpinGoGame
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent
from pokersim.ml.advanced_agents import PPOAgent, DeepCFRAgent, ImitationLearningAgent
from pokersim.game.state import GameState

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def create_opponent(player_id: int, opponent_type: str, device: torch.device, num_players: int) -> 'Agent':
    """
    Создает агента-противника указанного типа.

    Args:
        player_id (int): ID противника.
        opponent_type (str): Тип противника ('random', 'rule_based', 'ppo', 'deep_cfr', 'imitation').
        device (torch.device): Устройство для ML-агентов.
        num_players (int): Количество игроков в игре.

    Returns:
        Agent: Экземпляр агента-противника.

    Raises:
        ValueError: Если opponent_type недопустим.
    """
    logger.debug(f"Создание противника: player_id={player_id}, тип={opponent_type}")
    if opponent_type == 'random':
        return RandomAgent(player_id)
    elif opponent_type == 'rule_based':
        return RuleBased1Agent(player_id)
    elif opponent_type == 'ppo':
        return PPOAgent(
            player_id=player_id,
            game_state_class=GameState,
            num_players=num_players,
            device=device,
            epsilon=0.1,
            framework="pytorch"
        )
    elif opponent_type == 'deep_cfr':
        return DeepCFRAgent(
            player_id=player_id,
            game_state_class=GameState,
            num_players=num_players,
            device=device,
            epsilon=0.05,
            framework="pytorch"
        )
    elif opponent_type == 'imitation':
        input_dim = 2 * 52 + 5 * 52 + 1 + 3 * num_players + 3
        return ImitationLearningAgent(
            player_id=player_id,
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            action_dim=5,
            lr=0.001,
            device=device,
            expert=RuleBased1Agent(player_id + 100),  # Временный ID для эксперта
            batch_size=32
        )
    else:
        raise ValueError(f"Недопустимый тип противника: {opponent_type}")

def plot_metrics(episode_numbers: List[int], rewards: List[float], losses: List[float],
                 filename: str = '/kaggle/working/training_metrics.png') -> None:
    """
    Построение графиков метрик обучения (средняя награда и потери) и сохранение в файл.

    Args:
        episode_numbers (list): Список номеров эпизодов.
        rewards (list): Список средних наград.
        losses (list): Список средних потерь.
        filename (str): Путь для сохранения графика.
    """
    try:
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(episode_numbers, rewards, label='Средняя награда', color='blue')
        plt.title('Метрики обучения')
        plt.ylabel('Средняя награда')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(episode_numbers, losses, label='Средние потери', color='red')
        plt.xlabel('Эпизод')
        plt.ylabel('Средние потери')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logger.info(f"График метрик обучения сохранен в {filename}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении графика: {str(e)}")

def main(num_players: int = 3, epsilon: float = 0.1, num_episodes: int = 1, eval_interval: int = 1,
         device: Optional[torch.device] = None, opponent_types: List[str] = ['rule_based', 'random'],
         metrics_path: str = '/kaggle/working/training_metrics.png'):
    """
    Основная функция для обучения покерного ИИ.

    Args:
        num_players (int): Количество игроков.
        epsilon (float): Параметр исследования для агентов.
        num_episodes (int): Количество эпизодов обучения.
        eval_interval (int): Интервал для оценки и вывода метрик.
        device (torch.device): Устройство (CPU/GPU).
        opponent_types (List[str]): Типы противников.
    """
    try:
        # Инициализация устройства
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.init()  # Явная инициализация CUDA
        logger.info(f"Используемое устройство: {device}")

        input_dim = 2 * 52 + 5 * 52 + 1 + 3 * num_players + 3
        action_dim = 5

        num_opponents = num_players - 1
        if len(opponent_types) != num_opponents:
            logger.error(f"Ожидалось {num_opponents} типов противников, получено {len(opponent_types)}")
            raise ValueError(f"Количество типов противников должно быть равно num_players - 1")

        # Создание главного агента
        agent = PPOAgent(
            player_id=0,
            game_state_class=GameState,
            num_players=num_players,
            device=device,
            epsilon=epsilon,
            framework="pytorch"
        )
        # Создание противников
        opponents = [
            create_opponent(player_id=i+1, opponent_type=opp_type, device=device, num_players=num_players)
            for i, opp_type in enumerate(opponent_types)
        ]
        opponents_dict = {opp.player_id: opp for opp in opponents}
        logger.info(f"Инициализированы противники: {[opp.__class__.__name__ for opp in opponents]}")
        print(f"Инициализированы противники: {[opp.__class__.__name__ for opp in opponents]}")

        episode_numbers = []
        avg_rewards = []
        avg_losses = []

        logger.info(f"Начало обучения покерного ИИ на {num_episodes} эпизодов...")
        print(f"Начало обучения покерного ИИ на {num_episodes} эпизодов...")

        for episode in range(num_episodes):
            logger.debug(f"Начало эпизода {episode + 1}")
            game = SpinGoGame(num_players=num_players)
            total_reward = 0
            total_loss = 0
            hand_count = 0

            while not game.is_tournament_over():
                try:
                    state = game.start_new_hand()
                    if state is None:
                        logger.warning(f"Эпизод {episode + 1}: Не удалось начать новую раздачу")
                        break
                except ValueError as e:
                    logger.error(f"Эпизод {episode + 1}: Ошибка при начале раздачи: {e}")
                    break

                max_steps = 1000
                step = 0
                while not state.is_terminal() and step < max_steps:
                    try:
                        current_player = state.current_player
                        legal_actions = state.get_legal_actions()
                        if not legal_actions:
                            logger.warning(f"Эпизод {episode + 1}: Нет доступных действий для игрока {current_player}")
                            break

                        if current_player == agent.player_id:
                            action = agent.act(state)
                            logger.debug(f"Эпизод {episode + 1}: Агент выбрал действие: {action}")
                        else:
                            action = opponents_dict[current_player].act(state)
                            logger.debug(f"Эпизод {episode + 1}: Противник {current_player} выбрал действие: {action}")

                        if action not in legal_actions:
                            logger.error(f"Эпизод {episode + 1}: Недопустимое действие {action} от игрока {current_player}")
                            action = legal_actions[0]

                        state = state.apply_action(action)
                        game.current_game = state
                    except Exception as e:
                        logger.error(f"Эпизод {episode + 1}: Ошибка при выполнении действия: {str(e)}")
                        break

                    step += 1

                if step >= max_steps:
                    logger.warning(f"Эпизод {episode + 1}: Раздача завершена из-за превышения шагов")
                    continue

                try:
                    game.update_stacks_after_hand()
                    if game.current_game is not None:
                        payouts = game.current_game.get_payouts()
                        total_reward += payouts[agent.player_id]
                        hand_count += 1
                        logger.debug(f"Эпизод {episode + 1}: Выплата агенту: {payouts[agent.player_id]}, раздач: {hand_count}")
                    else:
                        logger.warning(f"Эпизод {episode + 1}: Отсутствует состояние игры для выплат")
                        continue
                except ValueError as e:
                    logger.error(f"Эпизод {episode + 1}: Ошибка при обновлении стеков: {e}")
                    continue

            try:
                logger.debug(f"Эпизод {episode + 1}: Сбор траекторий")
                trajectories = agent.ppo.collect_trajectories(num_trajectories=1)
                logger.debug(f"Эпизод {episode + 1}: Обновление политики")
                metrics = agent.ppo.update_policy(trajectories)
                total_loss = metrics.get('loss', 0)
                logger.debug(f"Эпизод {episode + 1}: Потери: {total_loss}")
            except Exception as e:
                logger.error(f"Эпизод {episode + 1}: Ошибка при обновлении политики: {str(e)}")
                continue

            avg_reward = total_reward / hand_count if hand_count > 0 else 0
            if (episode + 1) % eval_interval == 0:
                episode_numbers.append(episode + 1)
                avg_rewards.append(avg_reward)
                avg_losses.append(total_loss)
                metrics_message = (
                    f"Эпизод {episode + 1}/{num_episodes}: "
                    f"Средняя награда = {avg_reward:.2f}, Потери = {total_loss:.4f}, "
                    f"Раздач сыграно = {hand_count}, Противники = {[opp.__class__.__name__ for opp in opponents]}"
                )
                logger.info(metrics_message)
                print(metrics_message)
                plot_metrics(episode_numbers, avg_rewards, avg_losses, filename=metrics_path)

        logger.info("Обучение завершено")
        print("Обучение завершено")

    except Exception as e:
        logger.error(f"Неожиданная ошибка в main: {str(e)}")
        print(f"Ошибка: Неожиданная ошибка в main: {str(e)}")
        raise

if __name__ == "__main__":
    main()