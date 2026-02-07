# 🌌 Moltbook Galaxy: 에이전트 진화 시각화

Moltbook Galaxy는 Moltbook 생태계 내 인공지능 에인전트들의 영향력과 진화 과정을 매핑하기 위해 설계된 인터랙티브 네트워크 시각화 도구입니다. 에이전트들이 서로 상호작용하고, 영향력을 얻고, 시간이 지남에 따라 페르소나를 어떻게 변화시키는지 '타임랩스' 뷰로 보여줍니다.

![Moltbook Galaxy Screenshot](assets/preview.png)


<br/>

## 🚀 주요 기능

- **인터랙티브 맵**: 광활한 우주 형태의 에이전트 네트워크를 자유롭게 팬(Pan)하고 줌(Zoom)할 수 있습니다.

- **타임랩스 타임라인**: 하단 슬라이더를 통해 과거부터 현재(1월 28일 ~ 2월 7일)까지의 역사를 탐험하며 은하계의 진화를 관찰할 수 있습니다.

- **다이나믹 트랜지션**: 에이전트들이 영향력을 얻거나 소속이 바뀔 때 부드럽게 나타나고, 사라지고, 이동합니다.

- **에이전트 검색**: 특정 에이전트 이름을 검색하여 박동하는 하이라이트와 자동 줌 기능으로 빠르게 찾고 집중할 수 있습니다.

- **페르소나 필터링**: 개발자, 철학자, 신학자, 투자자 등 심리적 프로필에 따라 분류된 에이전트들을 확인할 수 있습니다.


<br/>

## 🛠 프로젝트 구조

- **`index.html`**: 메인 진입점입니다. HTML5 Canvas와 D3.js를 사용하여 시각화를 렌더링하는 순수 정적 HTML/JS 파일입니다.

- **`results/galaxy_history_data.js`**: 통합된 역사적 스냅샷 데이터가 포함된 핵심 데이터 엔진입니다.

- **`snapshot_tool/`**: 원본 Moltbook 데이터를 시각적 스냅샷으로 가공하는 Python 기반 파이프라인입니다.
  - `generate_evolution_snapshots.py`: 일일 좌표 및 페르소나 데이터를 생성합니다.
  - `consolidate_history.py`: 생성된 스냅샷들을 웹 최적화 JS 데이터 파일로 병합합니다.

- **`data/moltbook.db`**: 모든 에이전트 활동의 원천 데이터가 담긴 SQLite 데이터베이스입니다.


<br/>

## 💻 기술 스택

- **Frontend**: Vanilla JS, D3.js (줌/팬 로직), HTML5 Canvas (고성능 렌더링).

- **Backend Data Pipeline**: Python, Pandas, SQLAlchemy (SQLite), Scikit-Learn (좌표 생성을 위한 t-SNE).

- **Deployment**: GitHub Pages.


<br/>

## ⚖️ 라이선스

이 프로젝트는 개인적인 연구 및 시각화 목적으로 제작되었습니다. All rights reserved.
