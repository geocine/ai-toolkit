// Runs inside goja after the upstream bundle (entry.ts) set
// globalThis.__aitkUpstream. Mirrors what the upstream UI does on the new-job
// page: ui/src/extensions/modelArchs.ts (evaluating extension ui modules),
// ui/src/app/jobs/new/page.tsx (initial config, training folder, dataset) and
// SimpleJob.tsx (arch select via handleModelArchChange, "Add Dataset").
(function () {
  var U = globalThis.__aitkUpstream;

  // anything an extension module imports for rendering only (react, next/link,
  // form components) resolves to an inert stub: archs are plain data
  var stubHandler = {
    get: function (t, p) {
      if (p === '__esModule') return true;
      if (p === Symbol.toPrimitive) return function () { return ''; };
      return stub;
    },
    apply: function () { return stub; },
    construct: function () { return stub; },
  };
  var stub = new Proxy(function () {}, stubHandler);

  var SHIM = {
    react: stub,
    'react/jsx-runtime': stub,
    'react/jsx-dev-runtime': stub,
    'next/link': { __esModule: true, default: stub },
    '@/helpers/defaultSamples': U.defaultSamples,
    '@/components/formInputs': stub,
    '@/types': {},
    '@/app/jobs/new/options': U.options,
  };

  // JSON-safe copy (drops functions and React nodes)
  function plain(v) {
    if (v === undefined) return undefined;
    try {
      return JSON.parse(JSON.stringify(v));
    } catch (e) {
      return undefined;
    }
  }

  globalThis.__aitkLoadModules = function (modsJSON) {
    var mods = JSON.parse(modsJSON);
    var byName = new Map();
    var errors = [];
    mods.forEach(function (mod) {
      try {
        var module = { exports: {} };
        var require = function (name) {
          if (name in SHIM) return SHIM[name];
          throw new Error('cannot import "' + name + '" (available: ' + Object.keys(SHIM).join(', ') + ')');
        };
        new Function('require', 'module', 'exports', mod.code)(require, module, module.exports);
        var list = module.exports.AI_TOOLKIT_UI_MODELS;
        if (!Array.isArray(list)) throw new Error('must export an AI_TOOLKIT_UI_MODELS array');
        list.forEach(function (a, i) {
          if (!a || typeof a.name !== 'string' || typeof a.label !== 'string' || typeof a.group !== 'string') {
            throw new Error('AI_TOOLKIT_UI_MODELS[' + i + '] needs string name, label and group');
          }
        });
        list.forEach(function (a) {
          byName.set(a.name, { arch: a, source: mod.id });
        });
      } catch (e) {
        errors.push(mod.id + ': ' + ((e && e.message) || e));
      }
    });
    var entries = Array.from(byName.values()).sort(function (a, b) {
      return a.arch.label.localeCompare(b.arch.label, undefined, { sensitivity: 'base' });
    });
    globalThis.__aitkArchs = entries.map(function (e) { return e.arch; });
    return JSON.stringify({
      errors: errors,
      archs: entries.map(function (e) {
        var a = e.arch;
        return {
          name: a.name,
          label: a.label,
          group: a.group,
          source: e.source,
          defaults: plain(a.defaults) || {},
          disable_sections: plain(a.disableSections) || [],
          additional_sections: plain(a.additionalSections) || [],
          controls: plain(a.controls) || [],
          is_video_model: !!a.isVideoModel,
        };
      }),
    });
  };

  // page.tsx starts from objectCopy(migrateJobConfig(defaultJobConfig)); copy
  // first so the module default is never mutated
  function initialConfig() {
    return U.objectCopy(U.migrateJobConfig(U.objectCopy(U.defaultJobConfig)));
  }

  globalThis.__aitkDefaultConfig = function () {
    return JSON.stringify(initialConfig());
  };

  globalThis.__aitkRender = function (reqJSON) {
    var req = JSON.parse(reqJSON);
    var archs = globalThis.__aitkArchs || [];
    var cfg = initialConfig();
    var set = function (value, path) {
      cfg = U.setNestedValue(cfg, value, path);
    };

    if (req.training_folder) set(req.training_folder, 'config.process[0].training_folder');

    var current = cfg.config.process[0].model.arch;
    var archName = req.arch || current;
    if (archName !== current) {
      U.handleModelArchChange(archs, current, archName, cfg, set);
    }
    var arch = archs.find(function (a) { return a.name === archName; });

    // first dataset: page.tsx replaces the default folder path; more: the
    // "Add Dataset" button in SimpleJob.tsx
    (req.dataset_paths || []).forEach(function (p, i) {
      if (i === 0) {
        set(p, 'config.process[0].datasets[0].folder_path');
        return;
      }
      var nd = U.objectCopy(U.defaultDatasetConfig);
      nd.controls = (arch && arch.controls) || [];
      var defaults = (arch && arch.defaults) || {};
      for (var key in defaults) {
        var marker = 'datasets[x].';
        var idx = key.indexOf(marker);
        if (idx !== -1) nd[key.slice(idx + marker.length)] = defaults[key][0];
      }
      // the UI then has the user pick the folder
      nd.folder_path = p;
      set(cfg.config.process[0].datasets.concat([nd]), 'config.process[0].datasets');
    });

    if (req.name) set(req.name, 'config.name');

    // what selecting the arch produced, before overrides (which may change
    // model.arch on purpose, e.g. sd15 -> sd1)
    var applied = cfg.config.process[0].model.arch;
    var overrides = req.overrides || {};
    Object.keys(overrides).forEach(function (k) {
      set(overrides[k], k);
    });
    return JSON.stringify({ config: cfg, arch_applied: applied });
  };

  // ---- captioners: ui/src/helpers/captionOptions.ts and the dataset page's
  // CaptionDatasetModal (default caption job, captioner select via
  // handleCaptionerTypeChange, path_to_caption from the dataset)

  function opts(list) {
    return (plain(list) || []).map(function (o) { return { value: o.value, label: o.label }; });
  }

  globalThis.__aitkCaptioners = function () {
    var O = U.captionOptions;
    return JSON.stringify({
      default_config: plain(U.defaultCaptionJobConfig),
      captioners: (O.captionerTypes || []).map(function (c) {
        var defaults = {};
        var d = c.defaults || {};
        // [value when selected, value when switched away]
        Object.keys(d).forEach(function (k) {
          var v = plain(Array.isArray(d[k]) ? d[k][0] : d[k]);
          if (v !== undefined) defaults[k] = v;
        });
        return {
          name: c.name,
          label: c.label,
          group: c.group,
          defaults: defaults,
          model_options: opts(c.name_or_path_options),
          model2_options: opts(c.name_or_path2_options),
          caption_prompts: plain(c.captionPrompts) || {},
          additional_sections: plain(c.additionalSections) || [],
          supports_loras: !!c.supportsLoras,
          cloud_loras: plain(c.cloudLoras) || [],
          no_compile: !!c.noCompile,
          min_new_tokens: c.minNewTokens || null,
          multi_line_prompts: !!c.hasMultiLinePrompts,
        };
      }),
      options: {
        quantization: opts(O.quantizationOptions),
        caption_format: opts(O.captionFormatOptions),
        batch_size: opts(O.batchSizeOptions),
        max_res: opts(O.maxResOptions),
        max_new_tokens: opts(O.maxNewTokensOptions),
        video_fps: opts(O.videoFpsOptions),
        mtp_depth: opts(O.mtpDepthOptions),
      },
    });
  };

  globalThis.__aitkRenderCaption = function (reqJSON) {
    var req = JSON.parse(reqJSON);
    var cfg = U.objectCopy(U.defaultCaptionJobConfig);
    var set = function (value, path) {
      cfg = U.setNestedValue(cfg, value, path);
    };
    set(req.path, 'config.process[0].caption.path_to_caption');
    var current = cfg.config.process[0].type;
    if (req.captioner && req.captioner !== current) {
      U.handleCaptionerTypeChange(current, req.captioner, cfg, set);
    }
    if (req.name) set(req.name, 'config.name');
    var applied = cfg.config.process[0].type;
    var overrides = req.overrides || {};
    Object.keys(overrides).forEach(function (k) {
      set(overrides[k], k);
    });
    return JSON.stringify({ config: cfg, captioner_applied: applied });
  };
})();
