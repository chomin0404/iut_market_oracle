import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Spinner, Field, Button, ErrorBox, PageHeading } from "../components/ui";

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------

describe("Spinner", () => {
  it("renders a div with spin animation", () => {
    const { container } = render(<Spinner />);
    const el = container.firstChild as HTMLElement;
    expect(el.tagName).toBe("DIV");
    expect(el.style.animation).toMatch(/spin/);
  });

  it("accepts custom size and color props", () => {
    const { container } = render(<Spinner size={32} color="#ff0000" />);
    const el = container.firstChild as HTMLElement;
    expect(el.style.width).toBe("32px");
    expect(el.style.borderTopColor).toBe("rgb(255, 0, 0)");
  });
});

// ---------------------------------------------------------------------------
// Field
// ---------------------------------------------------------------------------

describe("Field", () => {
  it("renders as a <label> element", () => {
    const { container } = render(
      <Field label="Test Label">
        <input />
      </Field>
    );
    expect(container.querySelector("label")).toBeTruthy();
  });

  it("shows the label text", () => {
    render(
      <Field label="My Field">
        <input />
      </Field>
    );
    expect(screen.getByText("My Field")).toBeInTheDocument();
  });

  it("shows error message when error prop is provided", () => {
    render(
      <Field label="Name" error="This field is required">
        <input />
      </Field>
    );
    expect(screen.getByText("This field is required")).toBeInTheDocument();
  });

  it("does not render error element when error is absent", () => {
    render(
      <Field label="Name">
        <input />
      </Field>
    );
    expect(screen.queryByText(/required/i)).not.toBeInTheDocument();
  });

  it("wraps children inside the label (implicit association)", () => {
    const { container } = render(
      <Field label="Amount">
        <input data-testid="amount-input" />
      </Field>
    );
    const label = container.querySelector("label");
    expect(label?.querySelector("input")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

describe("Button", () => {
  it("renders children when not loading", () => {
    render(
      <Button accent="#4ade80" onClick={() => {}}>
        Run
      </Button>
    );
    expect(screen.getByText("Run")).toBeInTheDocument();
  });

  it("shows default loading label when loading=true", () => {
    render(
      <Button accent="#4ade80" loading>
        Run
      </Button>
    );
    expect(screen.getByText("Running…")).toBeInTheDocument();
  });

  it("shows custom loadingLabel when provided", () => {
    render(
      <Button accent="#4ade80" loading loadingLabel="Please wait…">
        Run
      </Button>
    );
    expect(screen.getByText("Please wait…")).toBeInTheDocument();
  });

  it("is disabled when loading", () => {
    render(
      <Button accent="#4ade80" loading>
        Run
      </Button>
    );
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("renders Spinner during loading", () => {
    const { container } = render(
      <Button accent="#4ade80" loading>
        Run
      </Button>
    );
    // Spinner renders a div with animation:spin
    const spinner = container.querySelector("div[style*='spin']");
    expect(spinner).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// ErrorBox
// ---------------------------------------------------------------------------

describe("ErrorBox", () => {
  it("prefixes message with 'Error:'", () => {
    render(<ErrorBox message="Something went wrong" />);
    expect(screen.getByText(/Error: Something went wrong/)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// PageHeading
// ---------------------------------------------------------------------------

describe("PageHeading", () => {
  it("renders heading text", () => {
    render(<PageHeading>My Page</PageHeading>);
    expect(screen.getByText("My Page")).toBeInTheDocument();
  });

  it("renders optional subtitle", () => {
    render(<PageHeading subtitle="Details here">My Page</PageHeading>);
    expect(screen.getByText("Details here")).toBeInTheDocument();
  });

  it("does not render subtitle element when absent", () => {
    render(<PageHeading>My Page</PageHeading>);
    expect(screen.queryByText("Details here")).not.toBeInTheDocument();
  });
});
