// Address autocomplete: fill the <datalist> from /api/suggest as the user types.
(function () {
  const input = document.getElementById("address");
  const list = document.getElementById("suggestions");
  const city = document.getElementById("city");
  if (!input || !list) return;
  let timer = null;
  let last = "";
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      if (q.length < 3 || q === last) return;
      last = q;
      try {
        const r = await fetch(`/api/suggest?city=${encodeURIComponent(city.value)}&q=${encodeURIComponent(q)}`);
        const items = await r.json();
        list.replaceChildren(...items.map((s) => Object.assign(document.createElement("option"), { value: s })));
      } catch (_) { /* autocomplete is optional */ }
    }, 200);
  });
})();
