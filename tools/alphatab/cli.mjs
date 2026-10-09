// alphaTex → 谱面模型 JSON / MIDI / Guitar Pro / SVG 片段（供 Python 端校验、合成与打印）
// 用法：node cli.mjs build <in.atex> <outDir> <name> [--svg-width 700]
// 成功时 stdout 最后一行是 {"json":..., "mid":..., "gp":..., "svg":...}
import * as alphaTab from '@coderline/alphatab';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

alphaTab.Logger.logLevel = alphaTab.LogLevel.None;

const PICK = { 0: null, 1: 'up', 2: 'down' };
const BRUSH = { 0: null, 1: 'up', 2: 'down', 3: 'up', 4: 'down' };
const FEEL = { 0: 'none', 1: 'triplet16th', 2: 'triplet8th', 3: 'dotted16th', 4: 'dotted8th', 5: 'scottish16th', 6: 'scottish8th' };

function toModel(score) {
  const staff = score.tracks[0].staves[0];
  const nStrings = staff.tuning.length;
  return {
    title: score.title,
    tempo: score.tempo,
    capo: staff.capo,
    tuning: staff.tuning,
    bars: staff.bars.map((bar) => {
      const mb = bar.masterBar;
      return {
        index: bar.index,
        start: mb.start,
        duration: mb.calculateDuration(),
        ts: [mb.timeSignatureNumerator, mb.timeSignatureDenominator],
        section: mb.section ? mb.section.text : null,
        triplet_feel: FEEL[mb.tripletFeel] ?? 'none',
        voices: bar.voices.map((voice) =>
          voice.beats
            .filter((b) => !b.isEmpty)
            .map((b) => ({
              start: b.displayStart,
              duration: b.displayDuration,
              tuplet: b.tupletNumerator > 0 ? b.tupletNumerator : 1,
              rest: b.isRest,
              dots: b.dots,
              chord: b.chord ? b.chord.name : null,
              pick: PICK[b.pickStroke] ?? null,
              brush: BRUSH[b.brushType] ?? null,
              notes: b.notes.map((n) => ({
                string: nStrings - n.string + 1,
                fret: n.fret,
                midi: n.realValue,
                tie: n.isTieDestination,
                dead: n.isDead,
              })),
            })),
        ),
      };
    }),
  };
}

function toMidi(score, settings) {
  const midiFile = new alphaTab.midi.MidiFile();
  const handler = new alphaTab.midi.AlphaSynthMidiFileHandler(midiFile, true);
  new alphaTab.midi.MidiFileGenerator(score, settings, handler).generate();
  return midiFile.toBinary();
}

function toSvg(score, width) {
  return new Promise((resolveSvg, reject) => {
    const settings = new alphaTab.Settings();
    settings.core.engine = 'svg';
    settings.core.enableLazyLoading = false;
    settings.core.useWorkers = false;
    const renderer = new alphaTab.rendering.ScoreRenderer(settings);
    renderer.width = width;
    const parts = [];
    renderer.partialRenderFinished.on((r) => parts.push(r.renderResult));
    renderer.renderFinished.on(() => resolveSvg(parts.filter((svg) => !svg.includes('rendered by alphaTab')).join('\n')));
    renderer.error.on(reject);
    renderer.renderScore(score, [0]);
  });
}

async function main() {
  const [cmd, inPath, outDir, name, ...rest] = process.argv.slice(2);
  if (cmd !== 'build' || !inPath || !outDir || !name) {
    console.error('usage: node cli.mjs build <in.atex> <outDir> <name> [--svg-width 700]');
    process.exit(64);
  }
  const wIdx = rest.indexOf('--svg-width');
  const svgWidth = wIdx >= 0 ? Number(rest[wIdx + 1]) : 700;
  const settings = new alphaTab.Settings();
  settings.core.logLevel = alphaTab.LogLevel.None;
  let score;
  try {
    score = alphaTab.importer.ScoreLoader.loadAlphaTex(readFileSync(inPath, 'utf8'), settings);
  } catch (e) {
    const inner = e.inner ?? e;
    const lines = [`alphaTex 解析失败: ${inPath}`, String(inner)];
    if (typeof inner.iterateDiagnostics === 'function') {
      for (const d of inner.iterateDiagnostics()) lines.push(`- ${d.message ?? JSON.stringify(d)}`);
    }
    console.error(lines.join('\n'));
    process.exit(2);
  }
  mkdirSync(outDir, { recursive: true });
  const out = {
    json: resolve(join(outDir, `${name}.json`)),
    mid: resolve(join(outDir, `${name}.mid`)),
    gp: resolve(join(outDir, `${name}.gp`)),
    svg: resolve(join(outDir, `${name}.svg.html`)),
  };
  writeFileSync(out.json, JSON.stringify(toModel(score)));
  writeFileSync(out.mid, toMidi(score, settings));
  writeFileSync(out.gp, new alphaTab.exporter.Gp7Exporter().export(score, settings));
  writeFileSync(out.svg, await toSvg(score, svgWidth));
  process.stdout.write('\n' + JSON.stringify(out) + '\n');
}

main().catch((e) => {
  console.error(String(e?.stack ?? e));
  process.exit(1);
});
