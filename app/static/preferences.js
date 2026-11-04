(() => {
  const key = 'dnascape.preferences.v1';
  if (window.dnascapePreferences) return;
  const storageError = () => emitEvent('dnascapePreferenceStorageError', {});
  window.dnascapePreferences = {
    load() {
      let preferences = null;
      let available = true;
      try {
        const text = localStorage.getItem(key);
        if (text && text.length <= 16384) {
          try { preferences = JSON.parse(text); } catch (_) { /* use defaults */ }
        }
      } catch (_) { available = false; }
      emitEvent('dnascapePreferences', {preferences, available});
    },
    save(preferences) {
      try { localStorage.setItem(key, JSON.stringify(preferences)); }
      catch (_) { storageError(); }
    },
    reset() {
      try { localStorage.removeItem(key); }
      catch (_) { storageError(); }
    },
  };
  // Other tabs in the same browser share only preferences, never workflow data.
  window.addEventListener('storage', event => {
    if (event.key === key && event.storageArea === localStorage) window.dnascapePreferences.load();
  });
})();
