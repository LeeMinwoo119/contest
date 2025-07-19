import numpy as np
import pandas as pd
import pickle
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromisingItemSelectionEnv(gym.Env):
    """
    유망품목 선별을 위한 강화학습 환경
    
    MDP 구성:
    - State: 5차원 벡터 [CAGR, RCA변화율, 점유율변화율, TSC, 수출단가변화율]
    - Action: 품목 인덱스 선택 (0~95)
    - Reward: 가중합 (0.3×CAGR + 0.3×RCA변화율 + 0.2×점유율변화율 + 0.15×TSC + 0.15×수출단가변화율)
    - Episode: 20회 선택 per episode
    """
    
    def __init__(self, 
                 max_selections: int = 20,
                 reward_weights: List[float] = [0.3, 0.3, 0.2, 0.15, 0.15],
                 allow_reselection: bool = False):
        """
        환경 초기화
        
        Args:
            max_selections: 에피소드당 최대 선택 횟수
            reward_weights: 보상 가중치 [CAGR, RCA변화율, 점유율변화율, TSC, 수출단가변화율]
            allow_reselection: 중복 선택 허용 여부
        """
        super(PromisingItemSelectionEnv, self).__init__()
        
        # 환경 설정
        self.max_selections = max_selections
        self.reward_weights = np.array(reward_weights)
        self.allow_reselection = allow_reselection
        
        # 데이터 로드
        self._load_data()
        
        # 액션 및 관측 공간 정의
        self.action_space = spaces.Discrete(self.num_items)  # 품목 선택 (0~95)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )  # 5차원 상태 공간 [0,1]
        
        # 에피소드 상태 초기화
        self.reset()
        
    def _load_data(self):
        """강화학습 데이터 로드"""
        try:
            # pickle 파일에서 RL 데이터 로드
            with open('rl_data.pkl', 'rb') as f:
                rl_data = pickle.load(f)
            
            # 상태 매트릭스 로드
            self.state_matrix = np.load('state_matrix.npy')
            
            # 품목 정보 로드
            self.item_info = pd.read_csv('normalized_indicators.csv')
            
            # 환경 파라미터 설정
            self.num_items = self.state_matrix.shape[0]  # 96개 품목
            self.state_dim = self.state_matrix.shape[1]   # 5차원 상태
            
            logger.info(f"데이터 로드 완료: {self.num_items}개 품목, {self.state_dim}차원 상태")
            
        except FileNotFoundError as e:
            logger.error(f"데이터 파일을 찾을 수 없습니다: {e}")
            raise
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        환경 초기화
        
        Args:
            seed: 랜덤 시드 (Gymnasium 표준)
            options: 추가 옵션 (Gymnasium 표준)
        
        Returns:
            초기 상태, 정보 딕셔너리
        """
        # 시드 설정
        if seed is not None:
            np.random.seed(seed)
        
        # 에피소드 상태 초기화
        self.current_step = 0
        self.selected_items = []
        self.cumulative_reward = 0.0
        self.episode_rewards = []
        
        # 선택 가능한 품목 초기화
        self.available_items = list(range(self.num_items))
        
        # 초기 상태 반환 (전체 품목의 평균 상태)
        initial_state = np.mean(self.state_matrix, axis=0)
        
        # 정보 딕셔너리
        info = {
            'available_items_count': len(self.available_items),
            'selected_items_count': 0,
            'cumulative_reward': 0.0
        }
        
        return initial_state.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        액션 수행
        
        Args:
            action: 선택할 품목 인덱스 (0~95)
            
        Returns:
            observation: 다음 상태
            reward: 보상
            terminated: 에피소드 자연 종료 여부
            truncated: 에피소드 강제 종료 여부
            info: 추가 정보
        """
        # 액션 유효성 검사
        if action not in self.available_items:
            if not self.allow_reselection:
                # 중복 선택 불가인 경우 낮은 보상
                reward = -0.1
                info = {
                    'reason': 'duplicate_selection',
                    'selected_item': action,
                    'item_name': self.item_info.iloc[action]['품목명']
                }
            else:
                # 중복 선택 허용인 경우 정상 처리
                reward = self._calculate_reward(action)
                info = {
                    'reason': 'reselection_allowed',
                    'selected_item': action,
                    'item_name': self.item_info.iloc[action]['품목명']
                }
        else:
            # 정상 선택
            reward = self._calculate_reward(action)
            info = {
                'reason': 'normal_selection',
                'selected_item': action,
                'item_name': self.item_info.iloc[action]['품목명']
            }
        
        # 선택된 품목 기록
        self.selected_items.append(action)
        self.episode_rewards.append(reward)
        self.cumulative_reward += reward
        
        # 선택된 품목을 사용 가능 목록에서 제거 (중복 선택 불가인 경우)
        if not self.allow_reselection and action in self.available_items:
            self.available_items.remove(action)
        
        # 스텝 증가
        self.current_step += 1
        
        # 에피소드 종료 조건 확인
        terminated = (self.current_step >= self.max_selections)  # 자연 종료 (목표 달성)
        truncated = (len(self.available_items) == 0)  # 강제 종료 (선택 불가능)
        
        # 다음 상태 계산
        if terminated or truncated:
            # 에피소드 종료 시 최종 상태
            next_state = np.zeros(self.state_dim)
        else:
            # 선택 가능한 품목들의 평균 상태
            if len(self.available_items) > 0:
                available_states = self.state_matrix[self.available_items]
                next_state = np.mean(available_states, axis=0)
            else:
                next_state = np.zeros(self.state_dim)
        
        # 추가 정보 업데이트
        info.update({
            'step': self.current_step,
            'cumulative_reward': self.cumulative_reward,
            'available_items_count': len(self.available_items),
            'selected_items_count': len(self.selected_items),
            'episode_average_reward': np.mean(self.episode_rewards)
        })
        
        return next_state.astype(np.float32), reward, terminated, truncated, info
    
    def _calculate_reward(self, action: int) -> float:
        """
        보상 계산
        
        Args:
            action: 선택된 품목 인덱스
            
        Returns:
            보상값
        """
        # 선택된 품목의 상태 벡터
        item_state = self.state_matrix[action]
        
        # 가중합 계산
        reward = np.dot(item_state, self.reward_weights)
        
        return float(reward)
    
    def get_item_info(self, action: int) -> Dict[str, Any]:
        """
        품목 정보 반환
        
        Args:
            action: 품목 인덱스
            
        Returns:
            품목 정보 딕셔너리
        """
        item_row = self.item_info.iloc[action]
        return {
            'hs_code': item_row['HS코드'],
            'item_name': item_row['품목명'],
            'cagr': item_row['CAGR'],
            'rca_change': item_row['RCA_변화율'],
            'share_change': item_row['점유율_변화율'],
            'tsc': item_row['TSC'],
            'price_change': item_row['수출단가_변화율'],
            'reward': self._calculate_reward(action)
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        에피소드 요약 정보 반환
        
        Returns:
            에피소드 요약 딕셔너리
        """
        if len(self.selected_items) == 0:
            return {"message": "아직 선택된 품목이 없습니다."}
        
        # 선택된 품목들의 정보
        selected_info = []
        for idx, item_idx in enumerate(self.selected_items):
            item_info = self.get_item_info(item_idx)
            item_info['step'] = idx + 1
            item_info['step_reward'] = self.episode_rewards[idx]
            selected_info.append(item_info)
        
        return {
            'total_steps': len(self.selected_items),
            'cumulative_reward': self.cumulative_reward,
            'average_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'selected_items': selected_info
        }
    
    def render(self, mode='human'):
        """환경 상태 출력"""
        if mode == 'human':
            print(f"\n=== 에피소드 상태 ===")
            print(f"현재 스텝: {self.current_step}/{self.max_selections}")
            print(f"선택된 품목 수: {len(self.selected_items)}")
            print(f"사용 가능한 품목 수: {len(self.available_items)}")
            print(f"누적 보상: {self.cumulative_reward:.4f}")
            
            if len(self.selected_items) > 0:
                print(f"최근 선택 품목: HS{self.item_info.iloc[self.selected_items[-1]]['HS코드']} - {self.item_info.iloc[self.selected_items[-1]]['품목명']}")
                print(f"최근 보상: {self.episode_rewards[-1]:.4f}")

def test_environment():
    """환경 테스트 함수"""
    print("=== 3단계: 강화학습 환경 구축 ===")
    print("=" * 50)
    
    # 환경 생성
    env = PromisingItemSelectionEnv(max_selections=5)  # 테스트용으로 5회만 선택
    
    print(f"1. 환경 초기화:")
    print(f"   - 품목 수: {env.num_items}")
    print(f"   - 상태 차원: {env.state_dim}")
    print(f"   - 액션 공간: {env.action_space}")
    print(f"   - 관측 공간: {env.observation_space}")
    
    # 환경 초기화
    initial_state, info = env.reset()
    print(f"\n2. 초기 상태: {initial_state}")
    print(f"   초기 정보: {info}")
    
    # 무작위 액션으로 테스트
    print(f"\n3. 무작위 액션 테스트:")
    for step in range(5):
        # 사용 가능한 품목 중 무작위 선택
        if len(env.available_items) > 0:
            action = np.random.choice(env.available_items)
        else:
            action = np.random.randint(0, env.num_items)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"   스텝 {step+1}: 액션={action}, 보상={reward:.4f}")
        print(f"   선택 품목: HS{info['selected_item']} - {info['item_name']}")
        print(f"   다음 상태: {next_state[:3]}... (처음 3개 차원)")
        
        if terminated or truncated:
            print(f"   에피소드 종료! (terminated={terminated}, truncated={truncated})")
            break
    
    # 에피소드 요약
    print(f"\n4. 에피소드 요약:")
    summary = env.get_episode_summary()
    print(f"   - 총 스텝: {summary['total_steps']}")
    print(f"   - 누적 보상: {summary['cumulative_reward']:.4f}")
    print(f"   - 평균 보상: {summary['average_reward']:.4f}")
    print(f"   - 최대 보상: {summary['max_reward']:.4f}")
    
    print(f"\n   선택된 품목 (상위 3개):")
    for i, item in enumerate(summary['selected_items'][:3]):
        print(f"   {i+1}. HS{item['hs_code']:02d}: {item['item_name'][:30]}... (보상: {item['step_reward']:.4f})")
    
    print(f"\n=== 3단계 완료 ===")
    print(f"강화학습 환경이 성공적으로 구축되었습니다!")
    print(f"다음 단계: 4단계 - PPO 에이전트 구현")

if __name__ == "__main__":
    test_environment() 