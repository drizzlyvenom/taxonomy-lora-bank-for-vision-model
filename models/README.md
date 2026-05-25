# Local Models

이 폴더는 로컬 teacher/backbone 모델을 놓기 위한 자리다.

대용량 모델 파일은 public GitHub repo에 커밋하지 않는다. 현재 `.gitignore`는 `*.gguf`,
`*.safetensors`, checkpoint 계열 파일을 제외한다.

모델을 연구 결과에 사용했다면 별도 manifest에 다음만 기록한다.

- 모델 이름
- 파일 해시
- 다운로드 출처
- 라이선스 확인 상태
- 사용한 역할: teacher, backbone, verifier 등

