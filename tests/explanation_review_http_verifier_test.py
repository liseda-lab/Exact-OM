"""Actual HTTPS verifier requires security headers only where runtime promises them."""

import httpx
import pytest

from tools.verify_review_environment import Journey, VerificationError, validate_response_csp

POLICY = "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
ORIGIN = "https://127.0.0.1:19184"


@pytest.mark.parametrize(
    "policy,label",
    [
        (None, "csp_present"),
        ("", "csp_present"),
        (POLICY.replace("object-src 'none'", "object-src *"), "csp_object-src"),
        (POLICY.replace("base-uri 'none'", ""), "csp_base-uri"),
        (POLICY + "; script-src 'unsafe-eval'", "csp_no_unsafe_eval"),
        (POLICY + "; default-src *", "csp_unique_directives"),
    ],
)
def test_missing_or_weakened_study_csp_cannot_pass_successful_http_response(policy, label):
    journey = Journey.__new__(Journey)
    journey.report = {"checks": []}
    headers = {} if policy is None else {"Content-Security-Policy": policy}
    with httpx.Client(
        base_url=ORIGIN,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={}, headers=headers)
        ),
    ) as client:
        journey.client = client
        with pytest.raises(VerificationError, match="probe:" + label):
            journey.request("GET", "/api/v1/study/state", label="probe")
    assert not journey.report["checks"]


def test_study_policy_required_but_exploration_api_has_no_invented_csp_contract():
    for origin, policy, expected in (
        (ORIGIN, POLICY, "study_response"),
        ("https://127.0.0.1:19185", None, "not_promised_by_exploration_api"),
    ):
        response = httpx.Response(
            200,
            json={},
            headers={} if policy is None else {"Content-Security-Policy": policy},
            request=httpx.Request("GET", origin + "/api/health"),
        )
        assert validate_response_csp(response, study_origin=ORIGIN, label="probe") == expected


@pytest.mark.parametrize("weaken", [False, True])
def test_study_html_requires_actual_strict_script_and_style_directives(weaken):
    policy = POLICY + "; script-src 'self' 'sha256-" + "a" * 43 + "='; style-src 'self'"
    policy += "; connect-src 'self'; font-src 'self'; form-action 'self'"
    if weaken:
        policy = policy.replace("style-src 'self'", "style-src 'self' 'unsafe-inline'")
    response = httpx.Response(
        200,
        headers={"Content-Security-Policy": policy, "Content-Type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", ORIGIN + "/participate/"),
    )
    if weaken:
        with pytest.raises(VerificationError, match="csp_document_styles"):
            validate_response_csp(response, study_origin=ORIGIN, label="probe")
    else:
        assert (
            validate_response_csp(response, study_origin=ORIGIN, label="probe") == "strict_document"
        )
