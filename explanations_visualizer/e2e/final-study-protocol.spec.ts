import { test, expect } from "@playwright/test";
import { harness, seedToTutorial, ParticipantApi } from "./v2Harness";

test.skip(!harness, "Requires the explicit final-study test fixture");

for (const journey of [1, 2]) {
  test(`confirmed skip, progress, navigation and selective methods · journey ${journey}`, async ({ page }) => {
    test.setTimeout(180000);
    const errors: string[] = [];
    page.on("pageerror", e => errors.push(e.message));
    // Background is API-seeded; the new skip and ranking interactions are real UI.
    const api = await seedToTutorial(page);
    await page.reload();
    await page.getByRole("button", { name: "Five short questions" }).click();
    const finish = page.getByRole("button", { name: "Finish the tutorial and start the scored cases" });
    await finish.click();
    const dialog = page.getByRole("dialog", { name: "Skip the rest of the tutorial?" });
    await dialog.getByRole("button", { name: "Return to tutorial", exact: true }).click();
    expect((await api.state()).stage).toBe("tutorial");
    await finish.click();
    await dialog.getByRole("button", { name: "Skip tutorial and start cases" }).click();
    await expect(page.getByRole("progressbar")).toHaveAttribute("max", "10");
    const skipped = await api.state();
    expect(skipped.tutorial_progress.completed_at).toBeNull();
    expect(skipped.tutorial_progress.skipped_at).toBeTruthy();
    expect(skipped.tutorial_progress.outstanding.length).toBeGreaterThan(0);
    const assignment = skipped.assignment_id;
    await page.getByLabel("Colour theme").selectOption("dark");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    expect((await api.state()).assignment_id).toBe(assignment);
    const seen = new Set<string>(), conditions: string[] = [];
    for (let n = 0; n < 10; n++) {
      await expect(page.getByRole("button", { name: "Keep initial order", exact: true })).toBeVisible();
      const current = await api.currentCase();
      expect(seen.has(current.case_id)).toBe(false); seen.add(current.case_id);
      conditions.push(current.condition);
      await expect(page.getByRole("progressbar")).toHaveAttribute("value", String(n));
      if (n === 0) {
        await expect(page.getByRole("button", { name: "Previous candidate", exact: true })).toBeDisabled();
        await page.getByRole("button", { name: "Next candidate", exact: true }).click();
        await expect(page.getByText("Candidate 2 of 5", { exact: true })).toBeVisible();
        await page.getByRole("button", { name: "Previous candidate", exact: true }).click();
        await page.setViewportSize({ width: 375, height: 900 });
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
        await page.setViewportSize({ width: 1440, height: 1000 });
      }
      if (current.condition === "ontology_baseline") {
        expect(current.workspace).toBeNull();
        expect(current.explanation_refs).toEqual([]);
      }
      await page.getByRole("button", { name: "Keep initial order", exact: true }).click();
      await page.getByRole("button", { name: "Submit answer", exact: true }).click();
      await expect.poll(async () => (await api.state()).stage).toBe(current.condition === "ontology_baseline" ? "consultation" : n === 9 ? "final" : "case");
      if (current.condition === "explanation") {
        // Consecutive explanation cases both use stage=case; wait for the actual
        // saved transition rather than accepting the preceding case's stage.
        await expect.poll(async () => (await api.state()).completed_cases).toBe(n + 1);
      }
      if (current.condition === "ontology_baseline") {
        const state = await api.state();
        await api.write("PUT", `/cases/${current.case_id}/consultation`, { presentation_id: current.presentation_id, form_version: state.forms.version, consultation_version: "consultation/3", consulted_external_ontologies: false, methods: [], other_method: null, resource_scope: null });
        await page.reload();
      }
    }
    expect(conditions.filter(c => c === "explanation")).toHaveLength(5);
    expect(conditions.filter(c => c === "ontology_baseline")).toHaveLength(5);
    expect((await api.state()).stage).toBe("final");
    expect(errors).toEqual([]);
    const state = await new ParticipantApi(page).state();
    await page.request.post(`${harness!.origin}/api/v1/admin/invitations/${state.session_id}/revoke`, { headers: { Authorization: `Bearer ${harness!.researcher_token}` } });
  });
}
