/**
 * The max_price_per_hour a launch sends for an offer quoted at `price`: 2 %
 * of headroom, rounded UP to the cent. The server refuses when the quote is
 * above the cap, and provider prices carry sub-cent parts (Vast's dph_total,
 * QuickPod's storage, RunPod's disk), so rounding to the nearest cent could
 * land below the quote for anything under $0.25/h and fail every time.
 */
export function priceCap(price: number): number {
  const cap = Math.ceil(price * 1.02 * 100 - 1e-9) / 100;
  return cap >= price ? cap : Math.ceil(price * 100) / 100 + 0.01;
}
