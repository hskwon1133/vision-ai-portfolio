const data = JSON.parse(document.getElementById("chart-data").textContent);

// 1) 동작별 예측 건수 (막대)
new Chart(document.getElementById("action-chart"), {
  type: "bar",
  data: {
    labels: data.actions.labels,
    datasets: [{ label: "예측 건수", data: data.actions.counts }],
  },
  options: { scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } },
});

// 2) 작업 상태별 건수 (원형)
new Chart(document.getElementById("status-chart"), {
  type: "pie",
  data: {
    labels: data.statuses.labels,
    datasets: [{ data: data.statuses.counts }],
  },
});
