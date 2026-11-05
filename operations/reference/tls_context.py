"""Process-local verified CA bundle; no repeated Windows certificate enumeration."""
import ssl

_context = None


def install(cafile=None):
    global _context
    if _context is None:
        if cafile is None:
            import certifi
            cafile = certifi.where()
        context = ssl.create_default_context(cafile=cafile)
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise RuntimeError('105_tls_verification_required')
        _context = context

    def verified_default(*args, **kwargs):
        if args or kwargs:
            raise RuntimeError('105_unexpected_tls_context_override')
        return _context

    # http.client.HTTPSConnection uses this factory when no explicit context is
    # supplied. This applies only inside the new AML control process.
    ssl._create_default_https_context = verified_default
    return _context
