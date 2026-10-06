// Fills the baseline tables from data/baseline.json (written by scripts/build_site.py from TUBE_BASELINE.json).
(async () => {
  const response = await fetch("data/baseline.json?v=__BUILD__");
  if (!response.ok) return;
  const data = await response.json();
  const um = v => (v === null || v === undefined) ? "–" : `${Math.round(v)} µm`;
  const names = {"0": "0, none", "1": "1, nominal", "2": "2, stress"};
  const table = document.getElementById("baseline-table");
  for (const level of ["0", "1", "2"]) {
    const s = data.summary[`level_${level}`];
    if (!s) continue;
    const row = table.insertRow();
    for (const [text, numeric] of [[names[level], false], [s.runs, true], [`${s.threads_placed} / ${s.sites_attempted}`, true],
                                   [um(s.placement_um_median), true], [um(s.placement_um_p90), true], [um(s.placement_um_max), true]]) {
      const cell = row.insertCell();
      cell.textContent = text;
      if (numeric) cell.className = "n";
    }
  }
  const sites = document.getElementById("site-table");
  const where = {0: "top of the dome", 5: "on the slope", 1: "on the slope"};
  for (const site of [0, 5, 1]) {
    const median = level => {
      const v = data.runs.filter(r => r.level === level).flatMap(r => r.sites).filter(x => x.site === site)
        .map(x => x.placement_lateral_um).sort((a, b) => a - b);
      return v.length ? v[Math.floor((v.length - 1) / 2)] / 2 + v[Math.ceil((v.length - 1) / 2)] / 2 : null;
    };
    const row = sites.insertRow();
    for (const text of [site, where[site], um(median(1)), um(median(2))]) row.insertCell().textContent = text;
  }
  const placed = Object.values(data.summary).reduce((a, s) => a + s.threads_placed, 0);
  const tried = Object.values(data.summary).reduce((a, s) => a + s.sites_attempted, 0);
  document.getElementById("placed").textContent = `${placed} / ${tried}`;
  document.getElementById("median1").textContent = um(data.summary.level_1.placement_um_median);
})();
