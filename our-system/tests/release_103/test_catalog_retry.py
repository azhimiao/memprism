import sqlite3
import pytest
from app.store import SqliteStore
from app import evidence_architecture as architecture

@pytest.fixture
def store(tmp_path):
    s=SqliteStore(tmp_path/'fixture.sqlite');s.initialize()
    yield s
    s._reads.close()
    with s._lock:s._writer.close()

def error(code):
    exc=sqlite3.DatabaseError('synthetic no source content')
    exc.sqlite_errorcode=code
    return exc

def test_corrupt_vtab_retries_once_on_new_read_handle(store,monkeypatch):
    seen=[]
    def assemble(db,*a,**k):
        seen.append(db)
        assert db.execute('SELECT 1').fetchone()[0]==1
        if len(seen)==1:raise error(sqlite3.SQLITE_CORRUPT_VTAB)
        return 'same_verified_evidence'
    monkeypatch.setattr(architecture,'assemble',assemble)
    assert store.assemble_catalog_evidence('synthetic-user',[],'alpha')=='same_verified_evidence'
    assert len(seen)==2 and seen[0] is not seen[1]
    assert store._reads.statistics()['created']==2

@pytest.mark.parametrize('code,attempts',[(sqlite3.SQLITE_CORRUPT_VTAB,2),(sqlite3.SQLITE_CORRUPT,1),(sqlite3.SQLITE_BUSY,1)])
def test_persistent_corruption_and_other_errors_fail_closed(store,monkeypatch,code,attempts):
    seen=[]
    def assemble(*a,**k):seen.append(1);raise error(code)
    monkeypatch.setattr(architecture,'assemble',assemble)
    with pytest.raises(sqlite3.DatabaseError):store.assemble_catalog_evidence('u',[],'alpha')
    assert len(seen)==attempts
