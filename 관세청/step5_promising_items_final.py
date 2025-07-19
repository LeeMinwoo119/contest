import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from step3_environment import PromisingItemSelectionEnv
from step4_ppo_agent import create_training_env
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromisingItemsExtractor:
    """
    유망품목 상위 5개 추출기
    """
    
    def __init__(self, model_path="ppo_promising_items_model"):
        """
        추출기 초기화
        """
        self.env = create_training_env()
        
        # 모델 로드 (에러 처리 포함)
        try:
            self.model = PPO.load(model_path)
            logger.info(f"모델 로드 성공: {model_path}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
        
        # 품목 정보 로드
        self.item_data = pd.read_csv('normalized_indicators.csv')
        
        # 보상 가중치 (환경과 동일)
        self.reward_weights = np.array([0.3, 0.3, 0.2, 0.15, 0.15])
        
        logger.info(f"추출기 초기화 완료: {len(self.item_data)}개 품목")
    
    def method1_direct_reward_calculation(self, top_k=5):
        """
        방법 1: 직접 보상 계산으로 상위 K개 추출 (수학적 최적해)
        """
        print(f"\n=== 방법 1: 직접 보상 계산 (수학적 최적해) ===")
        
        # 모든 품목의 보상 계산
        all_rewards = []
        for idx, row in self.item_data.iterrows():
            # 정규화된 지표 추출
            indicators = np.array([
                row['CAGR'], row['RCA_변화율'], row['점유율_변화율'], 
                row['TSC'], row['수출단가_변화율']
            ])
            
            # 보상 계산
            reward = np.dot(indicators, self.reward_weights)
            all_rewards.append({
                'hs_code': row['HS코드'],
                'item_name': row['품목명'],
                'reward': reward,
                'cagr': row['CAGR'],
                'rca_change': row['RCA_변화율'],
                'share_change': row['점유율_변화율'],
                'tsc': row['TSC'],
                'price_change': row['수출단가_변화율']
            })
        
        # 보상 기준 정렬
        all_rewards.sort(key=lambda x: x['reward'], reverse=True)
        
        # 상위 K개 선택
        top_items = all_rewards[:top_k]
        
        # 결과 출력
        print(f"상위 {top_k}개 품목:")
        for i, item in enumerate(top_items):
            print(f"{i+1}. HS{item['hs_code']:02d}: {item['item_name'][:50]}...")
            print(f"   보상: {item['reward']:.4f}")
            print(f"   지표: CAGR={item['cagr']:.3f}, RCA변화={item['rca_change']:.3f}, " +
                  f"점유율변화={item['share_change']:.3f}, TSC={item['tsc']:.3f}, " +
                  f"단가변화={item['price_change']:.3f}")
            print()
        
        return top_items
    
    def method2_policy_probability_single_step(self, top_k=5):
        """
        방법 2: 학습된 정책의 초기 확률 분포로 상위 K개 추출
        """
        print(f"\n=== 방법 2: 학습된 정책 확률 분포 ===")
        
        # 초기 상태에서 액션 확률 계산
        obs, info = self.env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # 정책 네트워크를 통한 액션 확률 계산
        with torch.no_grad():
            action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs
            action_probs = action_probs.cpu().numpy().flatten()
        
        # 확률 기준 정렬
        prob_rankings = []
        for idx, prob in enumerate(action_probs):
            item_row = self.item_data.iloc[idx]
            prob_rankings.append({
                'hs_code': item_row['HS코드'],
                'item_name': item_row['품목명'],
                'probability': prob,
                'reward': self.env._calculate_reward(idx),
                'cagr': item_row['CAGR'],
                'rca_change': item_row['RCA_변화율'],
                'share_change': item_row['점유율_변화율'],
                'tsc': item_row['TSC'],
                'price_change': item_row['수출단가_변화율']
            })
        
        # 확률 기준 정렬
        prob_rankings.sort(key=lambda x: x['probability'], reverse=True)
        
        # 상위 K개 선택
        top_items = prob_rankings[:top_k]
        
        # 결과 출력
        print(f"상위 {top_k}개 품목:")
        for i, item in enumerate(top_items):
            print(f"{i+1}. HS{item['hs_code']:02d}: {item['item_name'][:50]}...")
            print(f"   확률: {item['probability']:.4f}, 보상: {item['reward']:.4f}")
            print(f"   지표: CAGR={item['cagr']:.3f}, RCA변화={item['rca_change']:.3f}, " +
                  f"점유율변화={item['share_change']:.3f}, TSC={item['tsc']:.3f}, " +
                  f"단가변화={item['price_change']:.3f}")
            print()
        
        return top_items
    
    def method3_guided_selection(self, top_k=5):
        """
        방법 3: 가이드된 선택 - 매번 선택 가능한 품목 중 최선 선택
        """
        print(f"\n=== 방법 3: 가이드된 선택 (중복 제거) ===")
        
        obs, info = self.env.reset()
        selected_items = []
        
        for step in range(top_k):
            # 현재 선택 가능한 품목들
            available_items = self.env.available_items
            
            if not available_items:
                print(f"더 이상 선택 가능한 품목이 없습니다. (현재 {len(selected_items)}개 선택됨)")
                break
            
            # 선택 가능한 품목들 중에서 가장 높은 확률의 품목 선택
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs
                action_probs = action_probs.cpu().numpy().flatten()
            
            # 선택 가능한 품목들 중에서 가장 높은 확률 찾기
            best_action = None
            best_prob = -1
            
            for action in available_items:
                if action_probs[action] > best_prob:
                    best_prob = action_probs[action]
                    best_action = action
            
            # 선택 실행
            obs, reward, terminated, truncated, info = self.env.step(best_action)
            
            # 품목 정보 저장
            item_row = self.item_data.iloc[best_action]
            selected_items.append({
                'hs_code': item_row['HS코드'],
                'item_name': item_row['품목명'],
                'probability': best_prob,
                'reward': reward,
                'cagr': item_row['CAGR'],
                'rca_change': item_row['RCA_변화율'],
                'share_change': item_row['점유율_변화율'],
                'tsc': item_row['TSC'],
                'price_change': item_row['수출단가_변화율']
            })
            
            print(f"{step+1}. HS{item_row['HS코드']:02d}: {item_row['품목명'][:50]}...")
            print(f"   확률: {best_prob:.4f}, 보상: {reward:.4f}")
            print(f"   지표: CAGR={item_row['CAGR']:.3f}, RCA변화={item_row['RCA_변화율']:.3f}, " +
                  f"점유율변화={item_row['점유율_변화율']:.3f}, TSC={item_row['TSC']:.3f}, " +
                  f"단가변화={item_row['수출단가_변화율']:.3f}")
            print()
            
            if terminated or truncated:
                break
        
        return selected_items
    
    def method4_multiple_runs_consensus(self, top_k=5, n_runs=20):
        """
        방법 4: 다중 실행 합의 - 여러 번 실행해서 가장 많이 선택된 품목들
        """
        print(f"\n=== 방법 4: 다중 실행 합의 ({n_runs}회) ===")
        
        # 품목별 선택 횟수 추적
        selection_count = {}
        selection_rewards = {}
        
        for run in range(n_runs):
            obs, info = self.env.reset()
            run_selections = []
            
            for step in range(top_k):
                available_items = self.env.available_items
                
                if not available_items:
                    break
                
                # 확률적 선택 (다양성 확보)
                action, _ = self.model.predict(obs, deterministic=False)
                action = int(action)  # numpy 배열을 정수로 변환
                
                # 선택 가능한 품목인지 확인
                if action in available_items:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # 통계 업데이트
                    if action not in selection_count:
                        selection_count[action] = 0
                        selection_rewards[action] = []
                    
                    selection_count[action] += 1
                    selection_rewards[action].append(reward)
                    
                    run_selections.append(action)
                    
                    if terminated or truncated:
                        break
                else:
                    # 선택 불가능한 품목이면 가장 높은 확률 품목 선택
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs
                        action_probs = action_probs.cpu().numpy().flatten()
                    
                    # 선택 가능한 품목 중 최고 확률 선택
                    best_action = available_items[np.argmax([action_probs[a] for a in available_items])]
                    obs, reward, terminated, truncated, info = self.env.step(best_action)
                    
                    if best_action not in selection_count:
                        selection_count[best_action] = 0
                        selection_rewards[best_action] = []
                    
                    selection_count[best_action] += 1
                    selection_rewards[best_action].append(reward)
                    
                    if terminated or truncated:
                        break
        
        # 선택 횟수 기준 정렬
        consensus_items = []
        for item_idx, count in selection_count.items():
            item_row = self.item_data.iloc[item_idx]
            avg_reward = np.mean(selection_rewards[item_idx])
            
            consensus_items.append({
                'hs_code': item_row['HS코드'],
                'item_name': item_row['품목명'],
                'selection_count': count,
                'selection_frequency': count / n_runs,
                'avg_reward': avg_reward,
                'cagr': item_row['CAGR'],
                'rca_change': item_row['RCA_변화율'],
                'share_change': item_row['점유율_변화율'],
                'tsc': item_row['TSC'],
                'price_change': item_row['수출단가_변화율']
            })
        
        # 선택 횟수 기준 정렬
        consensus_items.sort(key=lambda x: x['selection_count'], reverse=True)
        
        # 상위 K개 선택
        top_items = consensus_items[:top_k]
        
        # 결과 출력
        print(f"상위 {top_k}개 품목:")
        for i, item in enumerate(top_items):
            print(f"{i+1}. HS{item['hs_code']:02d}: {item['item_name'][:50]}...")
            print(f"   선택 횟수: {item['selection_count']}/{n_runs} (빈도: {item['selection_frequency']:.3f})")
            print(f"   평균 보상: {item['avg_reward']:.4f}")
            print(f"   지표: CAGR={item['cagr']:.3f}, RCA변화={item['rca_change']:.3f}, " +
                  f"점유율변화={item['share_change']:.3f}, TSC={item['tsc']:.3f}, " +
                  f"단가변화={item['price_change']:.3f}")
            print()
        
        return top_items
    
    def extract_final_promising_items(self, top_k=5):
        """
        최종 유망품목 추출 - 4가지 방법 종합
        """
        print("="*80)
        print("🎯 최종 유망품목 추출 - 4가지 방법 종합 분석")
        print("="*80)
        
        # 4가지 방법으로 추출
        method1_results = self.method1_direct_reward_calculation(top_k)
        method2_results = self.method2_policy_probability_single_step(top_k)
        method3_results = self.method3_guided_selection(top_k)
        method4_results = self.method4_multiple_runs_consensus(top_k, n_runs=20)
        
        # 종합 분석
        print(f"\n🔍 종합 분석 및 최종 추천:")
        print("=" * 80)
        
        # 각 방법에서 나온 품목들 수집
        all_items = {}
        methods = [
            ("수학적 최적해", method1_results),
            ("정책 확률", method2_results),
            ("가이드된 선택", method3_results),
            ("다중 실행 합의", method4_results)
        ]
        
        for method_name, results in methods:
            for i, item in enumerate(results):
                hs_code = item['hs_code']
                if hs_code not in all_items:
                    all_items[hs_code] = {
                        'hs_code': hs_code,
                        'item_name': item['item_name'],
                        'methods': [method_name],
                        'ranks': [i + 1],
                        'avg_rank': i + 1,
                        'appearance_count': 1,
                        'total_score': top_k - i,  # 순위 기반 점수
                        'best_reward': item.get('reward', item.get('avg_reward', 0))
                    }
                else:
                    all_items[hs_code]['methods'].append(method_name)
                    all_items[hs_code]['ranks'].append(i + 1)
                    all_items[hs_code]['avg_rank'] = np.mean(all_items[hs_code]['ranks'])
                    all_items[hs_code]['appearance_count'] += 1
                    all_items[hs_code]['total_score'] += (top_k - i)
        
        # 종합 점수 기준 정렬 (출현 횟수 × 총 점수)
        final_ranking = sorted(all_items.values(), 
                              key=lambda x: (x['appearance_count'] * x['total_score']), 
                              reverse=True)
        
        # 최종 상위 5개 추천
        final_top5 = final_ranking[:top_k]
        
        print(f"📊 최종 추천 유망품목 상위 {top_k}개:")
        print("-" * 80)
        
        for i, item in enumerate(final_top5):
            print(f"{i+1}. HS{item['hs_code']:02d}: {item['item_name']}")
            print(f"   출현 횟수: {item['appearance_count']}/4 방법")
            print(f"   출현 방법: {', '.join(item['methods'])}")
            print(f"   평균 순위: {item['avg_rank']:.1f}")
            print(f"   종합 점수: {item['total_score']}")
            print(f"   최고 보상: {item['best_reward']:.4f}")
            print()
        
        return {
            'method1': method1_results,
            'method2': method2_results,  
            'method3': method3_results,
            'method4': method4_results,
            'final_top5': final_top5
        }

def main():
    """메인 실행 함수"""
    print("🎯 5단계: 유망품목 상위 5개 최종 추출")
    print("="*80)
    
    try:
        # 추출기 초기화
        extractor = PromisingItemsExtractor()
        
        # 최종 유망품목 추출
        results = extractor.extract_final_promising_items(top_k=5)
        
        print(f"\n✅ 유망품목 추출 완료!")
        print(f"🏆 최종 추천 유망품목:")
        print("=" * 50)
        
        for item in results['final_top5']:
            print(f"• HS{item['hs_code']:02d}: {item['item_name']}")
        
        print(f"\n🎉 프로젝트 완료!")
        print(f"강화학습 기반 유망품목 선별이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"유망품목 추출 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 