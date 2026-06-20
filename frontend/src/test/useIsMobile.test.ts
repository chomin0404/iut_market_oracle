import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useIsMobile } from "../hooks/useIsMobile";

// ---------------------------------------------------------------------------
// jsdom does not implement window.matchMedia — provide a minimal mock.
// ---------------------------------------------------------------------------

type MQListener = (e: MediaQueryListEvent) => void;

function makeMqMock(matches: boolean) {
  const listeners: MQListener[] = [];
  return {
    matches,
    addEventListener: (_: string, fn: MQListener) => listeners.push(fn),
    removeEventListener: (_: string, fn: MQListener) => {
      const idx = listeners.indexOf(fn);
      if (idx !== -1) listeners.splice(idx, 1);
    },
    // Helper to simulate a viewport change
    _trigger: (newMatches: boolean) => {
      listeners.forEach((fn) => fn({ matches: newMatches } as MediaQueryListEvent));
    },
  };
}

let mqMock: ReturnType<typeof makeMqMock>;

beforeEach(() => {
  mqMock = makeMqMock(false);
  vi.stubGlobal("matchMedia", vi.fn(() => mqMock));
  Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 1024 });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useIsMobile", () => {
  it("returns false on desktop (innerWidth >= 768)", () => {
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);
  });

  it("returns true when innerWidth < breakpoint", () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 375 });
    mqMock = makeMqMock(true);
    vi.stubGlobal("matchMedia", vi.fn(() => mqMock));
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(true);
  });

  it("updates when media query fires a change event", () => {
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);

    act(() => {
      mqMock._trigger(true);
    });

    expect(result.current).toBe(true);
  });

  it("accepts a custom breakpoint", () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 500 });
    mqMock = makeMqMock(true);
    vi.stubGlobal("matchMedia", vi.fn(() => mqMock));

    const { result } = renderHook(() => useIsMobile(600));
    expect(result.current).toBe(true);
  });

  it("removes event listener on unmount", () => {
    const removespy = vi.spyOn(mqMock, "removeEventListener");
    const { unmount } = renderHook(() => useIsMobile());
    unmount();
    expect(removespy).toHaveBeenCalledOnce();
  });
});
