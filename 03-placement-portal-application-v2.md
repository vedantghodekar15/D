# Project 3: Placement Portal Application V2

Repo: https://github.com/24f3003295/placement-portal-application-v2

> **Important:** I could not open this repository (GitHub rate-limited my requests, and a search did not surface it). So unlike files 1 and 2, this file is **not verified against your code**. The structure and the "typical" details below are a template. Before the interview, spend 15 minutes on the repo and fix every line marked **[CONFIRM]**. Do not state anything in the interview that your code does not do.

## 30-second pitch (adapt the details)

A web application that digitises campus placements. Three roles interact with it:

- **Admin (institute / placement cell):** approves companies, manages students and drives, monitors the whole process.
- **Company:** registers, gets approved, creates placement drives, reviews and shortlists applicants.
- **Student:** registers, builds a profile, browses drives, applies, and tracks application status.

The "V2" suggests this is the second iteration of an earlier version. **[CONFIRM]** what changed from V1 (for example a separate frontend, caching, background jobs, an API layer).

## The problem (product framing)

Placement is usually run over email and spreadsheets: eligibility is checked by hand, status updates are scattered, and nobody has one view of the funnel. The portal gives each role one place to work and gives the institute visibility.

- **Students:** one place to find drives and see where each application stands.
- **Companies:** structured applicant lists instead of email attachments.
- **Admin:** approval control and an overview of placement outcomes.

## Likely stack and structure [CONFIRM ALL]

Check the repo and delete what does not apply:

- Backend: Flask (Python) with SQLAlchemy and SQLite **[CONFIRM]**
- Frontend: Vue.js or Jinja templates **[CONFIRM]**
- Auth: token-based or session login with role-based access control **[CONFIRM]**
- Caching: Redis **[CONFIRM if present]**
- Background jobs: Celery for things like scheduled reminders, monthly reports, or CSV exports **[CONFIRM if present]**
- Database models to name: User, Student profile, Company profile, Drive/Job, Application **[CONFIRM exact models and relationships]**

## Core workflow to be able to narrate

1. Company registers, then waits for **admin approval**.
2. Approved company creates a **placement drive** with eligibility criteria and a deadline.
3. Admin reviews and approves the drive **[CONFIRM]**.
4. Students see approved drives and apply (one application per drive per student **[CONFIRM constraint]**).
5. Company reviews applicants and updates status (applied, shortlisted, selected, rejected **[CONFIRM statuses]**).
6. Student sees status changes; admin sees aggregate numbers.

## Design points interviewers like

- **Role-based access control:** each endpoint checks the user's role so students cannot reach admin actions. Explain exactly how yours is enforced **[CONFIRM]**.
- **Approval gates:** companies and drives need admin approval before going live, a simple governance control.
- **Data integrity:** unique constraint to prevent duplicate applications; foreign keys between users, drives and applications **[CONFIRM]**.
- **Status lifecycle:** applications move through defined states; this is the basis for funnel metrics.
- **Performance features (if you built them):** caching frequently read lists, background jobs for exports/reports, so the UI stays responsive.

## Product-analyst angle: the funnel and the metrics

This project is your strongest evidence for a product-analyst role because it is built around a funnel. Be ready to say what you would measure:

- **Funnel:** drives posted -> students viewing -> applications submitted -> shortlisted -> selected.
- **Conversion rates** at each stage, by branch, by company, by drive.
- **Time-to-hire:** days from drive creation to final selection.
- **Company side:** approval turnaround time, applicants per drive, response time to applicants.
- **Student side:** % of students with at least one application, average applications per student, placement rate.
- **Operational:** pending approvals, drives closing soon with no applicants.
- **Dashboards you would build for admin:** placement rate, offers by company, drop-off between stages.

Also prepare one data-driven insight: for example "if few students apply to drives with strict eligibility, relax or communicate criteria", and how you would test it.

## Likely interview questions

1. **Explain the project end to end.** Use the workflow above.
2. **Why three roles, and how are permissions enforced?**
3. **What changed in V2?** **[CONFIRM and prepare a clear answer]**
4. **How did you design the database?** Tables, relationships, constraints.
5. **What was hardest?** Pick a real one (RBAC, approval flow, status updates, frontend-backend integration).
6. **What would you improve?** Notifications, eligibility auto-checks, analytics dashboard, resume uploads, better search/filtering.
7. **How would you measure success of this product?** Use the funnel metrics above.
8. **How did you test it?** Say what you really did.

## 15-minute pre-interview checklist

- [ ] Open the README and note the exact stack and features
- [ ] List the database models and how they relate
- [ ] Run the app once and click through admin, company and student flows
- [ ] Write down 3 things that are different in V2
- [ ] Prepare one bug you fixed and one design decision you would change
- [ ] Remove the **[CONFIRM]** tags from this file once checked
