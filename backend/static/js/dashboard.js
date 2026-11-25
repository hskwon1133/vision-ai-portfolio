const data = JSON.parse(document.getElementById("chart-data").textContent);

// 6개 동작 + 판단 불확실: 예측이 0건인 동작도 막대에 표시
const ACTIONS = [
  ["bend_return", "허리 굽혔다 펴기"],
  ["leg_raise_lower", "다리 들었다 내리기"],
  ["walk_turn_walk", "걷다가 방향 바꾸기"],
  ["drink_return", "음료 마시고 돌아오기"],
  ["sit_stand", "앉았다 일어나기"],
  ["wave", "손 흔들기"],
  ["UNKNOWN", "판단 불확실"],
];
const STATUSES = [
  ["COMPLETED", "완료", "#1f8a5b"],
  ["PROCESSING", "처리 중", "#d69e2e"],
  ["PENDING", "대기", "#9aa3b2"],
  ["FAILED", "실패", "#c9362c"],
];

const countOf = (group, key) => {
  const i = group.labels.indexOf(key);
  return i === -1 ? 0 : group.counts[i];
};

// ----- 숫자 카드 -----
const total = data.statuses.counts.reduce((a, b) => a + b, 0);
document.getElementById("kpi-total").textContent = total;
document.getElementById("kpi-done").textContent = countOf(data.statuses, "COMPLETED");
document.getElementById("kpi-failed").textContent = countOf(data.statuses, "FAILED");
document.getElementById("kpi-preds").textContent = data.actions.counts.reduce((a, b) => a + b, 0);

// ----- 차트 공통 스타일 -----
Chart.defaults.font.family = '"Pretendard", -apple-system, "Segoe UI", "Malgun Gothic", sans-serif';
Chart.defaults.color = "#5b6475";

// 1) 동작별 예측 건수 (가로 막대)
new Chart(document.getElementById("action-chart"), {
  type: "bar",
  data: {
    labels: ACTIONS.map(([, ko]) => ko),
    datasets: [{
      label: "예측 건수",
      data: ACTIONS.map(([key]) => countOf(data.actions, key)),
      backgroundColor: ACTIONS.map(([key]) => (key === "UNKNOWN" ? "#e0b35a" : "#2f6fde")),
      borderRadius: 4,
      barThickness: 18,
    }],
  },
  options: {
    indexAxis: "y",
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (c) => ` ${c.parsed.x}건` } },
    },
    scales: {
      x: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: "#eef0f3" } },
      y: { grid: { display: false } },
    },
  },
});

// 2) 작업 상태별 건수 (도넛)
const statusCounts = STATUSES.map(([key]) => countOf(data.statuses, key));
new Chart(document.getElementById("status-chart"), {
  type: "doughnut",
  data: {
    labels: STATUSES.map(([, ko]) => ko),
    datasets: [{
      data: statusCounts,
      backgroundColor: STATUSES.map(([, , color]) => color),
      borderColor: "#ffffff",
      borderWidth: 2,
    }],
  },
  options: {
    maintainAspectRatio: false,
    cutout: "62%",
    plugins: {
      legend: {
        position: "right",
        labels: {
          usePointStyle: true,
          pointStyle: "circle",
          padding: 14,
          generateLabels: (chart) =>
            chart.data.labels.map((label, i) => ({
              text: `${label}  ${statusCounts[i]}건`,
              fillStyle: chart.data.datasets[0].backgroundColor[i],
              strokeStyle: "transparent",
              pointStyle: "circle",
              index: i,
            })),
        },
      },
      tooltip: {
        callbacks: {
          label: (c) => {
            const pct = total ? Math.round((c.parsed / total) * 100) : 0;
            return ` ${c.label}: ${c.parsed}건 (${pct}%)`;
          },
        },
      },
    },
  },
});
