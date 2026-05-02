// ACL 2023 Style Template for Typst
// Replicates the formatting of acl.sty

#let acl-paper(
  title: none,
  authors: (),
  abstract: none,
  body,
) = {
  // Page setup: A4 with 2.5cm margins
  set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
    numbering: "1",
    number-align: center,
  )

  // Base font: Times Roman, 11pt
  set text(font: "Times New Roman", size: 11pt, lang: "en")
  set par(justify: true, leading: 0.55em, first-line-indent: 1em)
  set heading(numbering: "1.1.1")

  // Heading styles
  show heading.where(level: 1): it => {
    set text(size: 12pt, weight: "bold")
    set par(first-line-indent: 0pt)
    v(2.0em)
    block(below: 1.5em, it)
  }

  show heading.where(level: 2): it => {
    set text(size: 11pt, weight: "bold")
    set par(first-line-indent: 0pt)
    v(1.8em)
    block(below: 0.8em, it)
  }

  show heading.where(level: 3): it => {
    set text(size: 11pt, weight: "bold")
    set par(first-line-indent: 0pt)
    v(1.5em)
    block(below: 0.5em, it)
  }

  // Code/monospace
  show raw.where(block: false): set text(font: ("Inconsolata", "Consolas", "Courier New"), size: 10pt)
  show raw.where(block: true): block.with(
    fill: luma(248),
    inset: 6pt,
    radius: 2pt,
    width: 100%,
  )
  show raw.where(block: true): set text(font: ("Inconsolata", "Consolas", "Courier New"), size: 9pt)

  // Link styling
  show link: set text(fill: rgb(0, 0, 128))

  // Figure caption styling
  show figure.caption: set text(size: 10pt)

  // Table styling
  set table(stroke: 0.5pt + black)

  // --- Title block ---
  {
    set par(first-line-indent: 0pt)
    align(center)[
      #v(0.5cm)
      #text(size: 15pt, weight: "bold")[#title]
      #v(0.8em)
      #for (i, author) in authors.enumerate() {
        if i > 0 { h(2em) }
        text(size: 12pt)[#author.name]
      }
      #v(0.3em)
      #for (i, author) in authors.enumerate() {
        if i > 0 { h(2em) }
        text(size: 11pt)[#author.affiliation]
      }
      #v(0.3em)
      #for (i, author) in authors.enumerate() {
        if i > 0 { h(2em) }
        text(size: 11pt, style: "italic")[#author.email]
      }
      #v(1.2em)
    ]
  }

  // --- Abstract ---
  {
    set par(first-line-indent: 0pt)
    pad(left: 0.6cm, right: 0.6cm)[
      #align(center)[
        #text(size: 12pt, weight: "bold")[Abstract]
      ]
      #v(0.5em)
      #set text(size: 10pt)
      #abstract
    ]
    v(1.2em)
  }

  // --- Two-column body ---
  columns(2, gutter: 0.6cm, body)
}
