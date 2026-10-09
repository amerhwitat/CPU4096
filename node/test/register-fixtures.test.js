import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { registerSnapshot } from '../src/index.js';

test('shared register snapshot vector matches the Node public API', async () => {
  const text = await readFile(new URL('../../contracts/fixtures/cpu4096-register-v1.tsv', import.meta.url), 'utf8');
  const rows = text.split(/\r?\n/).filter(line => line && !line.startsWith('#')).map(line => line.split('\t'));
  const row = rows.find(fields => fields[0] === 'snapshot');
  assert.ok(row, 'shared snapshot fixture missing');
  const [, width, lhs, , , expected] = row;
  const [expectedWidth, expectedWords] = expected.split(':');
  assert.deepEqual(registerSnapshot([BigInt(lhs) & ((1n << 64n) - 1n), BigInt(lhs) >> 64n]), {
    widthBits: Number(expectedWidth),
    words: expectedWords.split(',').map(word => BigInt(word).toString())
  });
  assert.equal(Number(width), Number(expectedWidth));
});
