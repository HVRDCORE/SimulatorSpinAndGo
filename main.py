"""
Основная точка входа для тренировки и симуляции покерного ИИ с настраиваемыми оппонентами,
оптимизированным логированием и визуализацией метрик.
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
from pokersim.config.config_manager import get_config

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Обработчик для консоли (INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Обработчик для файла (DEBUG)
file_handler = logging.FileHandler('poker_simulation.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Добавление обработчиков
logger.handlers = [console_handler, file_handler]

def create_opponent(player_id: int, opponent_type: str, device: torch.device, num_players: int) -> 'Agent':
    """
    Создаёт агента-оппонента указанного типа.
    """
    logger.debug(f"Создание оппонента: player_id={player_id}, тип={opponent_type}")
    config = get_config()
    if opponent_type == 'random':
        agent = RandomAgent(player_id)
    elif opponent_type == 'rule_based':
        agent = RuleBased1Agent(player_id)
    elif opponent_type == 'ppo':
        agent = PPOAgent(
            player_id=player_id,
            game_state_class=GameState,
            num_players=num_players,
            device=device,
            epsilon=config.get('training.epsilon', 0.1),
            framework="pytorch"
        )
    elif opponent_type == 'deep_cfr':
        agent = DeepCFRAgent(
            player_id=player_id,
            game_state_class=GameState,
            num_players=num_players,
            device=device,
            epsilon=config.get('training.epsilon', 0.05),
            framework="pytorch"
        )
    elif opponent_type == 'imLiberty':
        input_dim = 2 * 52 + 5 * 52 + 1 + 3 * num_players + 3
        agent = ImitationLearningAgent(
            player_id=player_id,
            input_dim=input_dim,
            hidden_dims=config.get('model.policy_network.hidden_layers', [128, 64, 32]),
            action_dim=config.get('model.policy_network.output_dim', 5),
            lr=config.get('training.learning_rate', 0.001),
            device=device,
            expert=RuleBased1Agent(player_id + 100),
            batch_size=config.get('training.batch_size', 32)
        )
    else:
        raise ValueError(f"Недопустимый тип оппонента: {opponent_type}")
    logger.info(f"Создан оппонент {agent.__class__.__name__} с player_id={player_id}")
    return agent

def plot_metrics(episode_numbers: List[int], rewards: List[float], losses: List[float],
                 filename: str = './output/training_metrics.png') -> None:
    """
    Построение графиков метрик обучения (средняя награда и потери) и сохранение в файл.
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
        logger.info(f"График метрик обучения сохранён в {filename}")
    except Exception as e:
        logger.error(f"Ошибка сохранения графика метрик: {str(e)}")

def main(num_players: int = 3, epsilon: float = 0.1, num_episodes: int = 1000, eval_interval: int = 100,
         device: Optional[torch.device] = None,
         opponent_types: List[str] = ['rule_based', 'random'],
         metrics_path: str = './output/training_metrics.png'):
    """
    Основная функция для тренировки покерного ИИ.
    """
    try:
        # Загрузка конфигурации
        try:
            config = get_config()
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise

        # Проверка конфигурации игроков
        config_num_players = config.get('game.table_size', num_players)
        if config_num_players != 3:
            logger.warning(f"Spin and Go требует 3 игроков, но конфигурация задаёт {config_num_players}. Используется 3.")
        num_players = 3
        epsilon = config.get('training.epsilon', epsilon)
        num_episodes = config.get('training.num_iterations', num_episodes)
        eval_interval = config.get('training.eval_frequency', eval_interval)

        if len(opponent_types) != num_players - 1:
            raise ValueError(f"Ожидалось {num_players - 1} типов оппонентов, получено {len(opponent_types)}")

        # Инициализация устройства
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используемое устройство: {device}")

        # Создание главного агента
        agent = PPOAgent(
            player_id=0,
            game_state_class=GameState,
            num_players=num_players,
            device=device,
            epsilon=epsilon,
            framework="pytorch"
        )
        # Создание оппонентов
        opponents = [
            create_opponent(player_id=i+1, opponent_type=opp_type, device=device, num_players=num_players)
            for i, opp_type in enumerate(opponent_types)
        ]
        opponents_dict = {opp.player_id: opp for opp in opponents}
        logger.info(f"Инициализированы оппоненты: {[opp.__class__.__name__ for opp in opponents]}")

        episode_numbers = []
        avg_rewards = []
        avg_losses = []

        logger.info(f"Начало тренировки покерного ИИ на {num_episodes} эпизодов...")

        for episode in range(num_episodes):
            logger.debug(f"Начало эпизода {episode + 1}")
            game = SpinGoGame(
                num_players=num_players,
                buy_in=config.get('tournament.buy_in', 10),
                starting_stack=config.get('tournament.starting_chips', 500)
            )
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
                    logger.error(f"Эпизод {episode + 1}: Ошибка начала раздачи: {e}")
                    break

                max_steps = 10000
                step = 0
                while not state.is_terminal() and step < max_steps:
                    try:
                        current_player = state.current_player
                        legal_actions = state.get_legal_actions()
                        if not legal_actions:
                            logger.warning(f"Эпизод {episode + 1}: Нет допустимых действий для игрока {current_player}")
                            break

                        logger.debug(f"Игрок {current_player} действует, состояние: {state}")
                        if current_player == agent.player_id:
                            action = agent.act(state)
                            logger.debug(f"Эпизод {episode + 1}: Агент выбрал действие: {action}")
                        else:
                            action = opponents_dict[current_player].act(state)
                            logger.debug(f"Эпизод {episode + 1}: Оппонент {current_player} выбрал действие: {action}")

                        if action not in legal_actions:
                            logger.error(
                                f"Эпизод {episode + 1}: Недопустимое действие {action} игроком {current_player}")
                            action = legal_actions[0]

                        state = state.apply_action(action)
                        game.current_game = state
                    except Exception as e:
                        logger.error(f"Эпизод {episode + 1}: Ошибка применения действия: {str(e)}")
                        break

                    step += 1

                if step >= max_steps:
                    logger.warning(
                        f"Эпизод {episode + 1}: Раздача завершена из-за превышения максимального количества шагов")
                    continue

                try:
                    if game.current_game is not None:
                        payouts = game.current_game.get_payouts()
                        total_reward += payouts[agent.player_id]
                        hand_count += 1
                        logger.debug(
                            f"Эпизод {episode + 1}: Выплата агенту: {payouts[agent.player_id]}, раздач: {hand_count}")
                        # Добавляем логирование стеков
                        logger.info(f"Эпизод {episode + 1}: Стеки после раздачи: {game.player_stacks}")
                    game.update_stacks_after_hand()
                except ValueError as e:
                    logger.error(f"Эпизод {episode + 1}: Ошибка обновления стеков: {e}")
                    continue

            try:
                logger.debug(f"Эпизод {episode + 1}: Сбор траекторий")
                trajectories = agent.ppo.collect_trajectories(num_trajectories=1)
                logger.debug(f"Эпизод {episode + 1}: Обновление политики")
                metrics = agent.ppo.update_policy(trajectories)
                total_loss = metrics.get('actor_loss', 0)
                logger.debug(f"Эпизод {episode + 1}: Потери: {total_loss}")
            except Exception as e:
                logger.error(f"Эпизод {episode + 1}: Ошибка обновления политики: {str(e)}")
                continue

            avg_reward = total_reward / hand_count if hand_count > 0 else 0
            if (episode + 1) % eval_interval == 0:
                episode_numbers.append(episode + 1)
                avg_rewards.append(avg_reward)
                avg_losses.append(total_loss)
                metrics_message = (
                    f"Эпизод {episode + 1}/{num_episodes}: "
                    f"Средняя награда = {avg_reward:.2f}, Потери = {total_loss:.4f}, "
                    f"Раздач сыграно = {hand_count}"
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