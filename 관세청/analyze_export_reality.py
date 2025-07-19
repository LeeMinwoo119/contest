import pandas as pd
import numpy as np

# 원본 데이터 로드
df = pd.read_csv('수출입 실적(품목별)_20250713.csv')
print("🔍 방법2 vs 방법4 결과의 실제 수출 데이터 분석")
print("=" * 60)

# 방법2 결과 분석
method2_codes = [97, 36, 81, 26, 83]
method4_codes = [97, 58, 74, 42, 71]

print("\n📊 방법 2 결과 분석 (정책 확률 분포)")
print("-" * 40)
for i, code in enumerate(method2_codes):
    items = df[df['HS코드'] == str(code).zfill(2)]
    if len(items) > 0:
        print(f"\n{i+1}. HS{code:02d}: {items.iloc[0]['품목명']}")
        
        # 2024년 데이터 찾기
        item_2024 = items[items['기간'] == '2024']
        item_2020 = items[items['기간'] == '2020']
        
        if len(item_2024) > 0 and len(item_2020) > 0:
            export_2024 = item_2024.iloc[0]['수출 금액']
            export_2020 = item_2020.iloc[0]['수출 금액']
            import_2024 = item_2024.iloc[0]['수입 금액']
            trade_balance = item_2024.iloc[0]['무역수지']
            
            print(f"   2024 수출: ${export_2024:,.0f}")
            print(f"   2024 수입: ${import_2024:,.0f}")
            print(f"   무역수지: ${trade_balance:,.0f}")
            print(f"   2020→2024 수출 증가율: {((export_2024/export_2020-1)*100):+.1f}%")

print("\n📊 방법 4 결과 분석 (다중 실행 합의)")
print("-" * 40)
for i, code in enumerate(method4_codes):
    items = df[df['HS코드'] == str(code).zfill(2)]
    if len(items) > 0:
        print(f"\n{i+1}. HS{code:02d}: {items.iloc[0]['품목명']}")
        
        # 2024년 데이터 찾기
        item_2024 = items[items['기간'] == '2024']
        item_2020 = items[items['기간'] == '2020']
        
        if len(item_2024) > 0 and len(item_2020) > 0:
            export_2024 = item_2024.iloc[0]['수출 금액']
            export_2020 = item_2020.iloc[0]['수출 금액']
            import_2024 = item_2024.iloc[0]['수입 금액']
            trade_balance = item_2024.iloc[0]['무역수지']
            
            print(f"   2024 수출: ${export_2024:,.0f}")
            print(f"   2024 수입: ${import_2024:,.0f}")
            print(f"   무역수지: ${trade_balance:,.0f}")
            print(f"   2020→2024 수출 증가율: {((export_2024/export_2020-1)*100):+.1f}%")

# 차이점 분석
print("\n🤔 왜 다른 결과가 나왔을까?")
print("=" * 60)
print("방법 2 (정책 확률 분포):")
print("- 단일 시점에서의 '이론적' 최적 선택")
print("- 모든 품목이 동등하게 선택 가능한 상황")
print("- 확률 차이가 매우 작음 (0.0185 vs 0.0167)")
print()
print("방법 4 (다중 실행 합의):")
print("- 실제 에피소드 진행 중 '동적' 선택")
print("- 이미 선택된 품목 제외 후 선택")
print("- 확률적 변동성 반영")
print()
print("🎯 결론: 방법4가 더 현실적!")
print("- 실제 포트폴리오 구성과 유사")
print("- 다양성 확보 (한 분야 집중 방지)")
print("- 확률적 정책의 탐험 특성 반영")

# 국가별 수출 관계 분석 (주요 품목 중심)
print("\n🌍 주요 유망품목의 국가별 수출 전략 분석")
print("=" * 60)

# 간단한 분석 (실제로는 더 자세한 국가별 데이터 필요)
print("HS97 예술품:")
print("- 중국, 일본, 유럽 → 한류 콘텐츠와 연계된 문화상품")
print("- 미국 → K-pop, 드라마 관련 굿즈, 전통 공예품")
print("- 고부가가치 틈새시장 공략")
print()
print("HS58 특수직물:")
print("- 동남아 → 의류 제조업체 대상 고급 소재")
print("- 유럽 → 패션 브랜드 대상 프리미엄 소재")
print("- 기술집약적 섬유 소재 경쟁력")
print()
print("HS74 구리:")
print("- 중국 → 전자산업, 건설업 원자재")
print("- 베트남, 인도 → 제조업 성장에 따른 수요 증가")
print("- 반도체, 전기차 산업 성장 수혜")
print()
print("HS42 가죽제품:")
print("- 유럽, 미국 → 프리미엄 가죽제품, 명품 브랜드")
print("- 중국 → 중산층 확대에 따른 럭셔리 소비 증가")
print("- K-브랜드 가죽제품 글로벌 진출")
print()
print("HS71 보석:")
print("- 중국, 홍콩 → 아시아 최대 보석 시장")
print("- 인도 → 전통적 보석 소비 대국")
print("- 미국, 유럽 → 디자인 보석, 맞춤 주얼리") 