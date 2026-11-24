const API = "/api/v1/analyses/";
const STATUS_TEXT = { PENDING: "대기", PROCESSING: "처리 중", COMPLETED: "완료", FAILED: "실패" };
const INPUT_TEXT = { UPLOAD: "영상 업로드", WEBCAM: "IP Webcam" };

const form = document.getElementById("analysis-form");
const csrfToken = form.querySelector("[name=csrfmiddlewaretoken]").value;
const startBtn = document.getElementById("start-btn");
const statusBox = document.getElementById("status-box");
let pollTimer = null;

// 1) 입력 방식에 따라 영상 칸 / 웹캠 안내 바꾸기
form.querySelectorAll("[name=input_type]").forEach((radio) => {
  radio.addEventListener("change", () => {
    const isWebcam = form.input_type.value === "WEBCAM";
    document.getElementById("video-field").hidden = isWebcam;
    document.getElementById("webcam-guide").hidden = !isWebcam;
  });
});

// 2) 분석 시작
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(form);
  const fileInput = form.querySelector("[name=video]");
  if (form.input_type.value === "WEBCAM" || fileInput.files.length === 0) {
    data.delete("video");
  }

  startBtn.disabled = true;
  setStatus("요청 중...", "");

  const res = await fetch(API, {
    method: "POST",
    headers: { "X-CSRFToken": csrfToken },
    body: data,
  });
  const body = await res.json();

  if (!res.ok) {
    // 서버가 돌려준 오류 이유를 그대로 보여줌
    const msg = Object.values(body).flat().join(" / ");
    setStatus(`요청 실패: ${msg}`, "FAILED");
    startBtn.disabled = false;
    return;
  }

  clearResult();
  watchJob(body.id);
  loadHistory();
});

// 3) 2초마다 상태 확인
function watchJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  const check = async () => {
    const res = await fetch(`${API}${jobId}/status/`);
    const s = await res.json();
    setStatus(`작업 #${jobId}: ${STATUS_TEXT[s.status] || s.status}`, s.status);
    if (s.is_done) {
      clearInterval(pollTimer);
      pollTimer = null;
      startBtn.disabled = false;
      loadDetail(jobId);
      loadHistory();
    }
  };
  check();
  pollTimer = setInterval(check, 2000);
}

// 4) 결과 표 그리기
async function loadDetail(jobId) {
  const res = await fetch(`${API}${jobId}/`);
  const job = await res.json();
  setStatus(
    `작업 #${job.id}: ${STATUS_TEXT[job.status] || job.status}` +
      (job.error_message ? ` — ${job.error_message}` : ""),
    job.status
  );

  const tbody = document.querySelector("#result-table tbody");
  tbody.innerHTML = "";
  if (job.predictions.length === 0) {
    addRow(tbody, ["-", "결과 없음", "-"]);
    return;
  }
  job.predictions.forEach((p) => {
    addRow(tbody, [p.track_id, p.action_label, p.confidence.toFixed(2)]);
  });
}

// 5) 최근 이력 (행을 누르면 그 작업 결과 보기)
async function loadHistory() {
  const res = await fetch(API);
  const jobs = await res.json();
  const tbody = document.querySelector("#history-table tbody");
  tbody.innerHTML = "";
  jobs.slice(0, 10).forEach((j) => {
    const tr = addRow(tbody, [
      j.id,
      j.title || "(제목 없음)",
      INPUT_TEXT[j.input_type] || j.input_type,
      STATUS_TEXT[j.status] || j.status,
      new Date(j.created_at).toLocaleString("ko-KR"),
    ]);
    tr.classList.add("clickable");
    tr.addEventListener("click", () => loadDetail(j.id));
  });
}

// --- 도우미 함수 ---
function setStatus(text, status) {
  statusBox.textContent = text;
  statusBox.className = `status ${status || ""}`;
}
function clearResult() {
  document.querySelector("#result-table tbody").innerHTML = "";
}
function addRow(tbody, cells) {
  const tr = document.createElement("tr");
  cells.forEach((c) => {
    const td = document.createElement("td");
    td.textContent = c;   // textContent: 사용자 입력을 HTML로 해석하지 않음 (보안)
    tr.appendChild(td);
  });
  tbody.appendChild(tr);
  return tr;
}

loadHistory();
