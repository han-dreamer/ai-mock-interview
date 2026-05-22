from app.models.jd import SkillItem, SkillMatrix
from app.models.resume import ProjectInfo, ResumeProfile
from app.resume.links import extract_resume_links
from app.resume.matcher import match_resume_to_jd
from app.resume.normalizer import normalize_resume_text


def test_normalize_resume_text_removes_page_noise():
    text, warnings = normalize_resume_text(
        "Name: Alex\n\n\nPage 1 of 2\nPython    FastAPI\n"
    )

    assert "Page 1 of 2" not in text
    assert "\n\n\n" not in text
    assert "Python FastAPI" in text
    assert warnings


def test_extract_resume_links_classifies_common_items():
    links = extract_resume_links(
        "GitHub: github.com/alex/agent\nBlog: https://juejin.cn/post/123\n"
        "Email: alex@example.com"
    )

    extracted = {(link.type, link.url) for link in links}
    assert ("github", "github.com/alex/agent") in extracted
    assert ("blog", "https://juejin.cn/post/123") in extracted
    assert ("email", "alex@example.com") in extracted


def test_match_resume_to_jd_maps_skills_to_projects():
    profile = ResumeProfile(
        name="Alex",
        skills=["Python", "FastAPI", "LangGraph"],
        projects=[
            ProjectInfo(
                name="AI Interview Agent",
                tech_stack=["FastAPI", "LangGraph"],
                description="A RAG-based mock interview app.",
                key_contributions=["Built the LangGraph interview flow."],
                potential_deep_dive_points=["State design", "RAG retrieval"],
            )
        ],
    )
    skill_matrix = SkillMatrix(
        position_title="AI App Developer",
        experience_level="junior",
        skills=[
            SkillItem(
                name="LangGraph",
                category="framework",
                weight=0.9,
                is_required=True,
            ),
            SkillItem(
                name="React",
                category="framework",
                weight=0.7,
                is_required=True,
            ),
        ],
    )

    match = match_resume_to_jd(profile, skill_matrix)

    assert match.matched_skills == ["LangGraph"]
    assert match.missing_skills == ["React"]
    assert match.relevant_projects == ["AI Interview Agent"]
    assert match.project_skill_map == {"AI Interview Agent": ["LangGraph"]}


def test_resume_profile_tolerates_null_llm_fields():
    profile = ResumeProfile.model_validate(
        {
            "name": None,
            "education": None,
            "skills": None,
            "projects": [
                {
                    "name": None,
                    "tech_stack": None,
                    "description": None,
                    "key_contributions": None,
                    "potential_deep_dive_points": None,
                }
            ],
            "experience": None,
            "highlights": None,
            "links": [{"url": "alex@example.com", "type": None}],
            "concerns": None,
            "parse_warnings": None,
            "summary": None,
        }
    )

    assert profile.name == ""
    assert profile.education == ""
    assert profile.skills == []
    assert profile.projects[0].name == ""
    assert profile.projects[0].tech_stack == []
    assert profile.links[0].type == "other"
    assert profile.summary == ""
