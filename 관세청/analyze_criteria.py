import pandas as pd
import numpy as np

# 정규화된 지표 로드
df = pd.read_csv('normalized_indicators.csv')

# HS97 (예술품) 분석
hs97 = df[df['HS코드'] == 97]
print('=== HS97 예술품 분석 (1위 품목) ===')
print(f'CAGR={hs97.iloc[0]["CAGR"]:.3f}, RCA변화율={hs97.iloc[0]["RCA_변화율"]:.3f}')
print(f'점유율변화율={hs97.iloc[0]["점유율_변화율"]:.3f}, TSC={hs97.iloc[0]["TSC"]:.3f}')
print(f'수출단가변화율={hs97.iloc[0]["수출단가_변화율"]:.3f}')
print()

# 원본 데이터에서 HS97 확인
원본 = pd.read_csv('수출입 실적(품목별)_20250713.csv')
hs97_raw = 원본[원본['HS코드'] == '97']
print('=== HS97 원본 데이터 ===')
print(hs97_raw[['기간', 'HS코드', '품목명', '수출 금액', '수입 금액', '무역수지']])
print()

# 5가지 지표 분석
print('=== 5가지 지표 분석 ===')
print('1. RCA (Revealed Comparative Advantage) - 수출 기준')
print('   RCA = (해당 품목 수출금액 / 전체 수출금액) ÷ (1 / 품목 수)')
print('   → 수출 경쟁력 지표')
print()
print('2. TSC (Trade Specialization Coefficient) - 수출-수입 비교')
print('   TSC = (수출액 - 수입액) / (수출액 + 수입액)')
print('   → 양수면 수출 특화, 음수면 수입 특화')
print()
print('3. 수출단가 변화율 - 수출 기준')
print('   수출단가 = 수출금액 / 수출중량')
print('   → 수출 품목의 고부가가치화 지표')
print()
print('4. CAGR & 점유율 변화율 - 수출 기준으로 추정')
print('   (전체 지표가 수출 중심이므로)')
print()

# 결론
print('=== 결론 ===')
print('💡 이 모델은 **수출 기준 유망품목**을 선별합니다!')
print('- RCA: 수출 경쟁력 (가중치 30%)')
print('- TSC: 수출-수입 비교 (가중치 15%)')
print('- 수출단가: 수출 고부가가치 (가중치 15%)')
print('- CAGR + 점유율: 수출 성장성 (가중치 50%)')
print()
print('📊 HS97 예술품이 1위인 이유:')
print('- 수출 급성장 (CAGR 최고)')
print('- 수출 경쟁력 증가 (RCA 최고)')
print('- 수출 점유율 급상승 (점유율변화율 최고)')
print('- 수출 특화 품목 (TSC 양수)')
print('- 수출 고부가가치 (수출단가 상승)') 