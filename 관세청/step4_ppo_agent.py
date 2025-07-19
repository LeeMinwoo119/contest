import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from step3_environment import PromisingItemSelectionEnv
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingCallback(BaseCallback):
    """
    학습 진행 상황을 모니터링하는 콜백
    """
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 에피소드 완료 시 보상 기록
        if len(self.locals['dones']) > 0 and self.locals['dones'][0]:
            if len(self.locals['infos']) > 0:
                episode_reward = self.locals['infos'][0].get('episode', {}).get('r', 0)
                episode_length = self.locals['infos'][0].get('episode', {}).get('l', 0)
                
                if episode_reward != 0:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    if len(self.episode_rewards) % 10 == 0:
                        recent_avg = np.mean(self.episode_rewards[-10:])
                        logger.info(f"에피소드 {len(self.episode_rewards)}: 최근 10회 평균 보상 = {recent_avg:.4f}")
        
        return True
    
    def get_statistics(self):
        """학습 통계 반환"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': np.min(self.episode_rewards) if self.episode_rewards else 0
        }

def create_training_env():
    """
    학습용 환경 생성
    """
    return PromisingItemSelectionEnv(
        max_selections=20,  # 실제 에피소드 길이
        reward_weights=[0.3, 0.3, 0.2, 0.15, 0.15],
        allow_reselection=False
    )

def train_ppo_agent(total_timesteps=50000, n_envs=4):
    """
    PPO 에이전트 학습
    
    Args:
        total_timesteps: 총 학습 스텝 수
        n_envs: 병렬 환경 수
    """
    print("=== 4단계: PPO 에이전트 구현 (Stable Baselines3) ===")
    print("=" * 60)
    
    # 벡터화된 환경 생성
    print("1. 학습 환경 생성...")
    env = make_vec_env(create_training_env, n_envs=n_envs)
    
    # PPO 모델 생성
    print("2. PPO 모델 초기화...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=dict(
            net_arch=[128, 128]  # 은닉층 구조
        ),
        verbose=1,
        seed=42
    )
    
    print(f"   - 정책 네트워크: MlpPolicy")
    print(f"   - 학습률: {model.learning_rate}")
    print(f"   - 배치 크기: {model.batch_size}")
    print(f"   - 에포크: {model.n_epochs}")
    print(f"   - 클리핑 범위: {model.clip_range}")
    print(f"   - 병렬 환경 수: {n_envs}")
    
    # 콜백 설정
    callback = TrainingCallback(verbose=1)
    
    # 로그 디렉토리 생성
    log_dir = "./ppo_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 학습 실행
    print(f"\n3. PPO 에이전트 학습 시작...")
    print(f"   - 총 학습 스텝: {total_timesteps:,}")
    print(f"   - 예상 에피소드 수: {total_timesteps // 20:,}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        tb_log_name="PPO_PromisingItems"
    )
    
    # 모델 저장
    model_path = "ppo_promising_items_model"
    model.save(model_path)
    print(f"\n4. 학습된 모델 저장: {model_path}")
    
    # 학습 통계
    stats = callback.get_statistics()
    print(f"\n5. 학습 통계:")
    print(f"   - 총 에피소드: {len(stats['episode_rewards'])}")
    print(f"   - 평균 보상: {stats['avg_reward']:.4f}")
    print(f"   - 최대 보상: {stats['max_reward']:.4f}")
    print(f"   - 최소 보상: {stats['min_reward']:.4f}")
    
    return model, stats

def test_trained_model(model_path="ppo_promising_items_model", n_test_episodes=5):
    """
    학습된 모델 테스트
    """
    print(f"\n=== 학습된 모델 테스트 ===")
    print("=" * 40)
    
    # 테스트 환경 생성
    env = create_training_env()
    
    # 모델 로드
    model = PPO.load(model_path)
    print(f"모델 로드 완료: {model_path}")
    
    # 테스트 에피소드 실행
    test_rewards = []
    test_selected_items = []
    
    for episode in range(n_test_episodes):
        obs, info = env.reset()  # Gymnasium API: tuple 반환
        episode_reward = 0
        selected_items = []
        
        print(f"\n테스트 에피소드 {episode + 1}:")
        
        while True:
            # 모델을 사용한 액션 예측 (obs만 전달)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            selected_items.append({
                'step': len(selected_items) + 1,
                'hs_code': info['selected_item'],
                'item_name': info['item_name'],
                'reward': reward
            })
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_selected_items.append(selected_items)
        
        # 상위 3개 선택 품목 출력
        top_items = sorted(selected_items, key=lambda x: x['reward'], reverse=True)[:3]
        print(f"  총 보상: {episode_reward:.4f}")
        print(f"  상위 3개 선택 품목:")
        for i, item in enumerate(top_items):
            print(f"    {i+1}. HS{item['hs_code']:02d}: {item['item_name'][:40]}... (보상: {item['reward']:.4f})")
    
    # 테스트 결과 요약
    print(f"\n테스트 결과 요약:")
    print(f"  - 평균 보상: {np.mean(test_rewards):.4f}")
    print(f"  - 최대 보상: {np.max(test_rewards):.4f}")
    print(f"  - 최소 보상: {np.min(test_rewards):.4f}")
    print(f"  - 표준편차: {np.std(test_rewards):.4f}")
    
    # 가장 성과가 좋은 에피소드의 전체 선택 품목 출력
    best_episode_idx = np.argmax(test_rewards)
    best_episode_items = test_selected_items[best_episode_idx]
    
    print(f"\n최고 성과 에피소드 ({best_episode_idx + 1}) 전체 선택 품목:")
    for item in best_episode_items:
        print(f"  {item['step']:2d}. HS{item['hs_code']:02d}: {item['item_name'][:50]}... (보상: {item['reward']:.4f})")
    
    return test_rewards, test_selected_items

def print_training_summary(stats):
    """
    학습 결과 요약 출력
    """
    if not stats['episode_rewards']:
        print("출력할 학습 데이터가 없습니다.")
        return
    
    print(f"\n=== 학습 결과 요약 ===")
    print(f"총 에피소드: {len(stats['episode_rewards'])}")
    print(f"평균 보상: {stats['avg_reward']:.4f}")
    print(f"최대 보상: {stats['max_reward']:.4f}")
    print(f"최소 보상: {stats['min_reward']:.4f}")
    
    # 최근 10개 에피소드 평균
    if len(stats['episode_rewards']) >= 10:
        recent_avg = np.mean(stats['episode_rewards'][-10:])
        print(f"최근 10개 에피소드 평균: {recent_avg:.4f}")
    
    # 학습 진행 상황 (간단한 텍스트 차트)
    if len(stats['episode_rewards']) > 20:
        print(f"\n보상 변화 추이 (최근 20개 에피소드):")
        recent_rewards = stats['episode_rewards'][-20:]
        for i, reward in enumerate(recent_rewards):
            bar_length = int(reward * 20)  # 보상을 20배 스케일로 변환
            bar = "█" * bar_length
            print(f"  {i+1:2d}: {reward:.3f} {bar}")
    
    print(f"\n💡 더 자세한 학습 모니터링은 텐서보드를 사용하세요:")
    print(f"   tensorboard --logdir=./ppo_tensorboard/")

def main():
    """
    메인 실행 함수
    """
    print("Stable Baselines3를 사용한 PPO 에이전트 구현")
    print("=" * 60)
    
    # 학습 실행
    model, stats = train_ppo_agent(total_timesteps=10000, n_envs=2)  # 테스트용 작은 값
    
    # 학습 결과 요약 출력
    print_training_summary(stats)
    
    # 학습된 모델 테스트
    test_rewards, test_items = test_trained_model()
    
    print(f"\n=== 4단계 완료 ===")
    print(f"PPO 에이전트 학습 및 테스트가 완료되었습니다!")
    print(f"다음 단계: 5단계 - 대규모 학습 및 최종 평가")

if __name__ == "__main__":
    main() 