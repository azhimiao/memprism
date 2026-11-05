"""Health describes serving controls; evidence validation stays in business IO."""
from contextlib import closing
import sqlite3

from fastapi.responses import JSONResponse
from app.config import Settings
from app.profiles import Profile
from app.serving_health_093 import ServingHealth
from deploy.runtime import check_runtime, configured_provider, PreflightFailure


def identity_preflight(path, profile):
    if not path.is_file():
        raise PreflightFailure('existing_local_working_database_required')
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=.5)) as db:
        db.execute('PRAGMA query_only=ON')
        row = db.execute('SELECT config_hash FROM k0_config WHERE singleton=1').fetchone()
        if row != (profile.config_hash,):
            raise PreflightFailure('database_identity_mismatch_rebuild_required')
    st = path.stat()
    return st.st_dev, st.st_ino


def build_app(settings=None, *, transport=None, verify_runtime=True):
    if verify_runtime:
        check_runtime()
    settings, provider = configured_provider(settings or Settings.from_env(), transport=transport)
    profile = Profile.build(settings, provider)
    checked = identity_preflight(settings.db_path, profile)
    from app import api
    previous = api._HealthReadiness
    api._HealthReadiness = ServingHealth
    try:
        application = api.create_app(settings, embedding_provider=provider)
    finally:
        api._HealthReadiness = previous
    application.state.retrieval.embedding_provider = provider.for_query()
    health = application.state.health_readiness
    if health.identity != checked:
        health.close()
        raise PreflightFailure('working_database_replaced_during_startup')
    # No startup global scan and no ever_ready bypass. Each original business
    # route still performs its own current user, source, lease and receipt checks.
    async def serving_health():
        result = await health.check()
        return JSONResponse(result, status_code=200 if result['status']=='ok' else 503,
                            headers={'Cache-Control':'no-store'})
    application.openapi_url = None
    application.router.routes = [r for r in application.router.routes
                                 if getattr(r, 'path', '') in {'/add','/search'}]
    application.add_api_route('/health', serving_health, methods=['GET'])
    return application


def create_app():
    try:
        return build_app()
    except (ValueError, sqlite3.Error, RuntimeError, OSError):
        raise PreflightFailure('local_093_serving_preflight_failed') from None
