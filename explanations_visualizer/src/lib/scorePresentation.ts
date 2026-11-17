/** Keep rounded endpoints distinct from recorded zero and one. No score is changed. */
export function formatScore(value: number): string {
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
  if ((Number(rounded) === 0 && value !== 0) || (rounded === "1.00" && value !== 1)) {
    return `≈${rounded}`;
  }
  return rounded;
}
