import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { NAV_ITEMS, Sidebar } from "../components/Sidebar";

// ---------------------------------------------------------------------------
// NAV_ITEMS structure
// ---------------------------------------------------------------------------

describe("NAV_ITEMS", () => {
  it("contains at least 5 navigation entries", () => {
    expect(NAV_ITEMS.length).toBeGreaterThanOrEqual(5);
  });

  it("first item routes to '/'", () => {
    expect(NAV_ITEMS[0].to).toBe("/");
  });

  it("every item has non-empty to, label, and icon", () => {
    for (const item of NAV_ITEMS) {
      expect(item.to.length).toBeGreaterThan(0);
      expect(item.label.length).toBeGreaterThan(0);
      expect(item.icon.length).toBeGreaterThan(0);
    }
  });

  it("includes Overview, Valuation, Bayesian, Entropy", () => {
    const labels = NAV_ITEMS.map((i) => i.label);
    expect(labels).toContain("Overview");
    expect(labels).toContain("Valuation");
    expect(labels).toContain("Bayesian");
    expect(labels).toContain("Entropy");
  });

  it("all paths are unique", () => {
    const paths = NAV_ITEMS.map((i) => i.to);
    expect(new Set(paths).size).toBe(paths.length);
  });
});

// ---------------------------------------------------------------------------
// Sidebar desktop rendering
// ---------------------------------------------------------------------------

describe("Sidebar (desktop)", () => {
  function renderSidebar(isOpen = true) {
    const onClose = vi.fn();
    const { container } = render(
      <MemoryRouter>
        <Sidebar isOpen={isOpen} onClose={onClose} isMobile={false} />
      </MemoryRouter>
    );
    return { container, onClose };
  }

  it("renders the IUT MARKET ORACLE brand text", () => {
    renderSidebar();
    expect(screen.getByText("IUT MARKET ORACLE")).toBeInTheDocument();
  });

  it("renders all nav item labels", () => {
    renderSidebar();
    for (const item of NAV_ITEMS) {
      expect(screen.getByText(item.label)).toBeInTheDocument();
    }
  });

  it("renders a nav element", () => {
    const { container } = renderSidebar();
    expect(container.querySelector("nav")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// Sidebar mobile — backdrop and Escape key
// ---------------------------------------------------------------------------

describe("Sidebar (mobile)", () => {
  function renderMobile(isOpen: boolean) {
    const onClose = vi.fn();
    render(
      <MemoryRouter>
        <Sidebar isOpen={isOpen} onClose={onClose} isMobile={true} />
      </MemoryRouter>
    );
    return { onClose };
  }

  it("renders backdrop when open on mobile", () => {
    const { container } = render(
      <MemoryRouter>
        <Sidebar isOpen={true} onClose={vi.fn()} isMobile={true} />
      </MemoryRouter>
    );
    // Backdrop is a fixed div that covers the screen
    const fixedDivs = Array.from(container.querySelectorAll("div")).filter(
      (d) => d.style.position === "fixed" && d.style.background.includes("rgba")
    );
    expect(fixedDivs.length).toBeGreaterThanOrEqual(1);
  });

  it("calls onClose when Escape key is pressed", () => {
    const { onClose } = renderMobile(true);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("does not call onClose for other keys", () => {
    const { onClose } = renderMobile(true);
    fireEvent.keyDown(document, { key: "Enter" });
    expect(onClose).not.toHaveBeenCalled();
  });
});
