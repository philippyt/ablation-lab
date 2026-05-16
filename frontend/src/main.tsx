import { render } from "preact";
import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("no #root in document");
render(<App />, root);
