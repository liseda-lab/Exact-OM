"use client";

import { useState } from "react";
import { Dialog } from "@/components/common/Dialog";

const EXAMPLE = `Prefix(:=<https://example.org/inspection#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
Ontology(<https://example.org/inspection>
 Declaration(Class(:Container))
 Declaration(Class(:Crate))
 AnnotationAssertion(rdfs:label :Container "container")
 AnnotationAssertion(rdfs:label :Crate "crate")
 AnnotationAssertion(rdfs:comment :Crate "A box used to carry or store things.")
 SubClassOf(:Crate :Container)
)`;

function downloadExample() {
  const url = URL.createObjectURL(new Blob([EXAMPLE], { type: "text/plain" }));
  const link = document.createElement("a");
  link.href = url; link.download = "inspection-practice.ofn";
  document.body.appendChild(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Optional reference guides. No assessment, completion requirement or scored-case effects. */
export function InspectionGuides() {
  const [guide, setGuide] = useState<"protege" | "text" | null>(null);
  const [step, setStep] = useState(0);
  const [answer, setAnswer] = useState(false);
  const close = () => { setGuide(null); setStep(0); setAnswer(false); };
  const title = guide === "protege" ? "Inspecting an ontology in Protégé" : "Inspecting an ontology in a text editor";
  return <aside className="inspection-guides" aria-label="Optional ontology inspection guides">
    <div><strong>New to inspecting ontologies?</strong><p className="meta">Two short guides you can open or skip.</p></div>
    <div className="study-actions">
      <button type="button" className="btn btn-sm" onClick={() => setGuide("protege")}>Protégé guide</button>
      <button type="button" className="btn btn-sm" onClick={() => setGuide("text")}>Text editor guide</button>
    </div>
    {guide && <Dialog title={title} onClose={close} closeLabel="Skip guide" wide>
      <div className="inspection-guide-content">
        <p className="meta">Optional · {step + 1} of 3 · fictional practice file</p>
        {step === 0 ? <>
          <h3>What is in an ontology file?</h3>
          <p>An ontology describes concepts and their relationships. A class is a type of thing, such as a crate. Its IRI is its unique identifier. Labels give readable names, and annotations may provide descriptions.</p>
          <figure className="guide-figure">
            <svg viewBox="0 0 640 170" role="img" aria-label="Crate is a subclass of Container. A crate is a kind of container.">
              <rect x="20" y="40" width="220" height="80" rx="12" />
              <text x="130" y="87" textAnchor="middle">Crate</text>
              <path d="M240 80 H400 M388 70 L400 80 L388 90" />
              <text x="320" y="55" textAnchor="middle" className="guide-svg-small">is a kind of</text>
              <rect x="400" y="40" width="220" height="80" rx="12" />
              <text x="510" y="87" textAnchor="middle">Container</text>
            </svg>
            <figcaption>Container is a parent of Crate. The two concepts do not have the same meaning.</figcaption>
          </figure>
          <button type="button" className="btn" onClick={downloadExample}>Download the tiny practice ontology</button>
          <p className="meta">Use this file to try the guide. Use the two study files for the ranking task.</p>
        </> : guide === "protege" ? step === 1 ? <>
          <h3>Open the file and find a class</h3>
          <ol className="plain-steps">
            <li>Open Protégé Desktop. Choose <strong>File → Open</strong> and select <strong>inspection-practice.ofn</strong>.</li>
            <li>Open the <strong>Entities</strong> tab and select <strong>Classes</strong>. Expand the class hierarchy to find <strong>crate</strong>. You can also use the search field.</li>
            <li>Select crate. Look at its <strong>Annotations</strong> and <strong>Description</strong>, including its parent class. You do not need to start a reasoner for this.</li>
          </ol>
          <figure className="guide-figure guide-editor">
            <div className="guide-editor-heading">Entities → Classes</div>
            <div className="guide-editor-columns"><div>▾ owl:Thing<br />　▾ container<br />　　<strong>crate</strong></div><div><strong>Annotations</strong><p>label · crate</p><p>comment · A box used to carry or store things.</p><strong>SubClass Of</strong><p>container</p></div></div>
            <figcaption>Simplified illustration of the class views. The layout may differ in your version.</figcaption>
          </figure>
          <p className="meta">Need to install it? <a href="https://protege.stanford.edu/software/" target="_blank" rel="noreferrer">Download Protégé Desktop</a>.</p>
        </> : <>
          <h3>Try a quick inspection</h3><p>Find crate in the practice file. What is its parent? Where can you read its description?</p>
          <button type="button" className="btn" onClick={() => setAnswer(true)}>Show answer</button>
          {answer && <p className="note">Its parent is container. The description is the comment in Annotations.</p>}
          <p>For a study case, open the supplied ontology, then search for the concept’s label or the identifier shown next to it. Check its description and parents before comparing it with the candidate in the other ontology.</p>
        </> : step === 1 ? <>
          <h3>Find the concept in the file</h3>
          <ol className="plain-steps">
            <li>Open <strong>inspection-practice.ofn</strong> in a plain-text editor, such as VS Code, Notepad++ on Windows, or TextEdit in plain-text mode on macOS.</li>
            <li>Use <strong>Find</strong> to search for <code>Crate</code>. Move through the matches to see its label, comment and parent.</li>
            <li>The study files use OWL Functional Syntax. Each statement starts with its type, followed by values in parentheses. Statements may span several lines.</li>
          </ol>
          <figure className="guide-figure">
            <pre className="guide-code"><code>{'AnnotationAssertion(rdfs:label :Crate "crate")\nAnnotationAssertion(rdfs:comment :Crate\n  "A box used to carry or store things.")\nSubClassOf(:Crate :Container)'}</code></pre>
            <figcaption>The first two statements give a label and comment. The last says every crate is a container.</figcaption>
          </figure>
        </> : <>
          <h3>Try a quick inspection</h3><p>Search for <code>SubClassOf</code> in the practice file. Which class is the parent of Crate?</p>
          <button type="button" className="btn" onClick={() => setAnswer(true)}>Show answer</button>
          {answer && <p className="note">Container. In <code>SubClassOf(:Crate :Container)</code>, the more specific class comes first.</p>}
          <p>For a study case, copy its IRI and search in the corresponding ontology file. If the full IRI is not present, search its final identifier, such as C21750 or DOID_0112183. Prefixes at the top of the file may shorten IRIs.</p>
          <p>The study files are much larger than this example. If your editor struggles to open one, use an editor that supports large files or Protégé. Read the files without changing them.</p>
        </>}
        <div className="study-actions">
          <button type="button" className="btn btn-quiet" onClick={close}>Skip guide</button>
          {step > 0 && <button type="button" className="btn" onClick={() => { setStep(n => n - 1); setAnswer(false); }}>Back</button>}
          <button type="button" className="btn btn-primary" onClick={() => { if (step === 2) close(); else { setStep(n => n + 1); setAnswer(false); } }}>{step === 2 ? "Done" : "Next"}</button>
        </div>
      </div>
    </Dialog>}
  </aside>;
}
