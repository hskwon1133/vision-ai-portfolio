const API = "/api/v1/analyses/";
const STATUS_TEXT = { PENDING: "대기", PROCESSING: "처리 중", COMPLETED: "완료", FAILED: "실패" };
const INPUT_TEXT = { UPLOAD: "영상 업로드", WEBCAM: "IP Webcam" };
const ACTION_TEXT = {
  bend_return: "허리 굽혔다 펴기",
  leg_raise_lower: "다리 들었다 내리기",
  walk_turn_walk: "걷다가 방향 바꾸기",
  drink_return: "음료 마시고 돌아오기",
  sit_stand: "앉았다 일어나기",
  wave: "손 흔들기",
  UNKNOWN: "판단 불확실",
};

const form = document.getElementById("analysis-form");
const csrfToken = form.querySelector("[name=csrfmiddlewaretoken]").value;
const startBtn = document.getElementById("start-btn");
const statusBox = document.getElementById("status-box");
const resultList = document.getElementById("result-list");
const fileInput = form.querySelector("[name=video]");
const dropzone = document.getElementById("dropzone");
const dzText = document.getElementById("dz-text");
let pollTimer = null;
let currentJobId = null;

// ----- 입력 방식 전환 -----
form.querySelectorAll("[name=input_type]").forEach((radio) => {
  radio.addEventListener("change", () => {
    const isWebcam = form.input_type.value === "WEBCAM";
    document.getElementById("video-field").hidden = isWebcam;
    document.getElementById("webcam-guide").hidden = !isWebcam;
  });
});

// ----- 파일 선택 표시 -----
fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  dropzone.classList.toggle("has-file", !!f);
  dzText.textContent = f ? `${f.name} (${(f.size / 1024 / 1024).toFixed(1)}MB)` : "파일을 선택하거나 여기로 끌어다 놓으세요";
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, () => dropzone.classList.add("dragover"))
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, () => dropzone.classList.remove("dragover"))
);

// ----- 분석 시작 -----
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(form);
  if (form.input_type.value === "WEBCAM" || fileInput.files.length === 0) {
    data.delete("video");
  }

  startBtn.disabled = true;
  startBtn.textContent = "요청 중...";
  setStatus("요청을 보내는 중입니다...", "");
  setSteps(null);

  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "X-CSRFToken": csrfToken },
      body: data,
    });
    const body = await res.json();

    if (!res.ok) {
      const msg = Object.values(body).flat().join(" / ");
      setStatus(`요청 실패: ${msg}`, "FAILED");
      resetButton();
      return;
    }

    clearResult();
    watchJob(body.id);
    loadHistory();
  } catch (err) {
    setStatus("서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.", "FAILED");
    resetButton();
  }
});

// ----- 2초마다 상태 확인 -----
function watchJob(jobId) {
  currentJobId = jobId;
  if (pollTimer) clearInterval(pollTimer);
  startBtn.textContent = "분석 중...";

  const check = async () => {
    const res = await fetch(`${API}${jobId}/status/`);
    const s = await res.json();
    const extra = s.status === "PROCESSING" && form.input_type.value === "WEBCAM"
      ? " — 카메라 앞에서 동작해 주세요" : "";
    setStatus(`작업 #${jobId} ${STATUS_TEXT[s.status] || s.status}${extra}`, s.status);
    setSteps(s.status);

    if (s.is_done) {
      clearInterval(pollTimer);
      pollTimer = null;
      resetButton();
      loadDetail(jobId);
      loadHistory();
    }
  };
  check();
  pollTimer = setInterval(check, 2000);
}

// ----- 결과 그리기 -----
async function loadDetail(jobId) {
  currentJobId = jobId;
  const res = await fetch(`${API}${jobId}/`);
  const job = await res.json();

  setStatus(
    `작업 #${job.id} ${STATUS_TEXT[job.status] || job.status}` +
      (job.error_message ? ` — ${job.error_message}` : ""),
    job.status
  );
  setSteps(job.status);
  highlightHistory(job.id);

  resultList.innerHTML = "";
  if (job.predictions.length === 0) {
    resultList.appendChild(el("p", "empty", job.status === "FAILED" ? "분석에 실패해 결과가 없습니다." : "결과가 없습니다."));
    return;
  }
  job.predictions.forEach((p) => resultList.appendChild(resultCard(p)));
}

function resultCard(p) {
  const isUnknown = p.action_label === "UNKNOWN";
  const card = el("div", `result${isUnknown ? " unknown" : ""}`);

  card.appendChild(el("div", "result-track", `#${p.track_id}`));

  const label = el("div", "result-label", ACTION_TEXT[p.action_label] || p.action_label);
  if (!isUnknown) label.appendChild(el("span", "result-code", p.action_label));
  card.appendChild(label);

  card.appendChild(el("div", "result-conf", `${Math.round(p.confidence * 100)}%`));

  const bar = el("div", "bar");
  const fill = el("span");
  fill.style.width = `${Math.max(2, p.confidence * 100)}%`;
  bar.appendChild(fill);
  card.appendChild(bar);
  return card;
}

// ----- 최근 이력 -----
async function loadHistory() {
  const res = await fetch(API);
  const jobs = await res.json();
  const tbody = document.querySelector("#history-table tbody");
  tbody.innerHTML = "";

  if (jobs.length === 0) {
    const tr = document.createElement("tr");
    const td = el("td", "empty", "아직 분석 기록이 없습니다.");
    td.colSpan = 5;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  jobs.slice(0, 10).forEach((j) => {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    tr.dataset.id = j.id;
    tr.appendChild(el("td", "num", `#${j.id}`));
    tr.appendChild(el("td", "", j.title || "(제목 없음)"));
    tr.appendChild(el("td", "", INPUT_TEXT[j.input_type] || j.input_type));
    const statusTd = el("td");
    statusTd.appendChild(el("span", `badge badge-${j.status}`, STATUS_TEXT[j.status] || j.status));
    tr.appendChild(statusTd);
    tr.appendChild(el("td", "muted", new Date(j.created_at).toLocaleString("ko-KR")));
    tr.addEventListener("click", () => loadDetail(j.id));
    tbody.appendChild(tr);
  });
  highlightHistory(currentJobId);
}

// ----- 도우미 -----
function setStatus(text, status) {
  statusBox.textContent = text;
  statusBox.className = `status ${status || ""}`;
}

function setSteps(status) {
  const items = document.querySelectorAll("#steps li");
  items.forEach((li) => (li.className = ""));
  document.getElementById("done-label").textContent = status === "FAILED" ? "실패" : "완료";
  if (!status) return;

  const [pending, processing, done] = items;
  if (status === "PENDING") {
    pending.className = "active";
  } else if (status === "PROCESSING") {
    pending.className = "done";
    processing.className = "active";
  } else if (status === "COMPLETED") {
    pending.className = processing.className = done.className = "done";
  } else if (status === "FAILED") {
    pending.className = processing.className = "done";
    done.className = "failed";
  }
}

function highlightHistory(jobId) {
  document.querySelectorAll("#history-table tbody tr").forEach((tr) => {
    tr.classList.toggle("selected", String(tr.dataset.id) === String(jobId));
  });
}

function clearResult() {
  resultList.innerHTML = "";
  resultList.appendChild(el("p", "empty", "분석이 끝나면 결과가 표시됩니다."));
}

function resetButton() {
  startBtn.disabled = false;
  startBtn.textContent = "AI 분석 시작";
}

// textContent만 사용: 사용자 입력이 HTML로 해석되지 않음 (보안)
function el(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

loadHistory();
