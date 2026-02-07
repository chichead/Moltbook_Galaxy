# 몰트북 갤럭시 배포 가이드 (Moltbook Galaxy Deployment Guide)

이 가이드는 **몰트북 갤럭시(Moltbook Galaxy)** 시각화 프로젝트를 **Railway** 또는 **Render**와 같은 클라우드 서비스(PaaS)를 통해 웹에 배포하는 방법을 설명합니다.

## 중요: 폴더 구조 설정 (Root Directory)

이 프로젝트는 `galaxy`라는 하위 폴더 안에 배포할 어플리케이션이 들어있습니다.
따라서 배포 서비스 설정에서 반드시 **Root Directory**를 지정해야 합니다.

*   **Railway**: Settings -> General -> **Root Directory**를 `/galaxy`로 설정하세요.
*   **Render**: 설정 화면에서 **Root Directory** 항목에 `galaxy`라고 입력하세요.

## 준비 사항

1.  **GitHub 계정**: 코드가 GitHub에 업로드되어 있어야 합니다.
2.  **서비스 계정**: [Railway](https://railway.app/) 또는 [Render](https://render.com/) 가입.

---

## 옵션 1: Railway로 배포하기 (추천)

1.  **로그인**: [railway.app](https://railway.app/) 접속 & 로그인.
2.  **새 프로젝트**: **"New Project"** -> **"Deploy from GitHub repo"**.
3.  **레포지토리 선택**: `moltbook` 선택.
4.  **설정 변경 (중요)**:
    *   배포가 시작되기 전에, 혹은 실패했다면 해당 프로젝트의 **Settings**로 들어갑니다.
    *   **Root Directory** 항목을 찾아 `/galaxy`로 변경하고 저장합니다.
    *   그러면 자동으로 다시 빌드와 배포가 시작됩니다.
5.  **주소 생성**: **Settings** -> **Networking** -> **Generate Domain** 클릭.

---

## 옵션 2: Render로 배포하기

1.  **로그인**: [render.com](https://render.com/) 접속.
2.  **새 웹 서비스**: **"New +"** -> **"Web Service"**.
3.  **GitHub 연결**: `moltbook` 리포지토리 선택.
4.  **설정 입력**:
    *   **Name**: `moltbook-galaxy`
    *   **Root Directory**: `galaxy`  <-- **(필수)**
    *   **Runtime**: **Python 3**
    *   **Build Command**: `pip install -r requirements.txt` (그냥 이렇게 입력하면 됩니다, galaxy 폴더 기준이므로)
    *   **Start Command**: `gunicorn app.main:app -k uvicorn.workers.UvicornWorker` 또는 Railway처럼 `Procfile` 자동 감지.
5.  **서비스 생성**: 배포 시작.

---

## 데이터 업데이트 (SQLite)

*   데이터는 `galaxy/data/moltbook.db`에 저장됩니다.
*   새로운 데이터를 반영하려면 로컬에서 업데이트 후 `git push`를 하시면 됩니다.
*   서버가 재시작되면 서버 상의 임시 데이터는 초기화되므로, 꼭 로컬에서 DB를 관리해서 올리는 것을 권장합니다.
