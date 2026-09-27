(() => {
    const navbar = document.querySelector('.navbar');
    const toggle = document.querySelector('.nav-menu-toggle');
    const nav = document.querySelector('.nav-links');
    const more = document.querySelector('.nav-more');
    const moreToggle = document.querySelector('.nav-more-toggle');

    if (!navbar || !toggle || !nav) return;

    const setMenu = (open) => {
        navbar.classList.toggle('nav-open', open);
        toggle.setAttribute('aria-expanded', String(open));
        toggle.textContent = open ? '✕' : '☰';
    };

    const setMore = (open) => {
        if (!more || !moreToggle) return;
        more.classList.toggle('open', open);
        moreToggle.setAttribute('aria-expanded', String(open));
    };

    toggle.addEventListener('click', () => setMenu(!navbar.classList.contains('nav-open')));

    moreToggle?.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        setMore(!more.classList.contains('open'));
    });

    nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
        setMore(false);
        setMenu(false);
    }));

    document.addEventListener('click', event => {
        if (more && !more.contains(event.target)) setMore(false);
        if (navbar.classList.contains('nav-open') && !navbar.contains(event.target)) setMenu(false);
    });

    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
            setMore(false);
            setMenu(false);
        }
    });

    window.addEventListener('resize', () => {
        if (window.innerWidth > 1000) setMenu(false);
    });

    const currentPath = (window.location.pathname === '/' ? '/index.html' : window.location.pathname).replace(/\\/g, '/').toLowerCase();
    const currentHash = window.location.hash || '';
    const matchesCurrent = (link) => {
        const target = new URL(link.getAttribute('href') || '', window.location.href);
        const targetPath = target.pathname.replace(/\\/g, '/').toLowerCase();
        if (targetPath !== currentPath) return false;
        if (target.hash) return target.hash === currentHash;
        return currentHash === '';
    };

    nav.querySelectorAll('a[data-nav]').forEach(link => {
        if (matchesCurrent(link)) link.classList.add('active');
    });

    const activeMore = [...(more?.querySelectorAll('a') || [])].some(matchesCurrent);
    if (activeMore) moreToggle?.classList.add('active');

    const activeMain = nav.querySelector('a.active');
    if (activeMain) activeMain.setAttribute('aria-current', 'page');

})();
