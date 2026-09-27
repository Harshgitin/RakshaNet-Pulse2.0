import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.11.2/dist/maplibre-gl.mjs';

(() => {
    const INDIA_CENTER = [79.0, 22.8];
    const OPENFREEMAP_STYLE = 'https://tiles.openfreemap.org/styles/bright';
    const OSM_RASTER_STYLE = {
        version: 8,
        sources: {
            osm: {
                type: 'raster',
                tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                tileSize: 256,
                attribution: '&copy; OpenStreetMap contributors'
            }
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
    };

    const state = {
        overview: null,
        events: [],
        map: null,
        mapStyleFallback: false,
        mapEvents: [],
        mapReady: false,
        mapPointerHandlersBound: false,
        filterBusy: false
    };

    const $ = id => document.getElementById(id);
    const esc = value => String(value ?? '')
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;').replace(/'/g,'&#039;');

    function dateOf(value) {
        if (!value) return null;
        const d = new Date(value);
        return Number.isFinite(d.getTime()) ? d : null;
    }
    function fmt(value) {
        const d = dateOf(value);
        return d ? d.toLocaleString('en-IN',{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:true}) : 'Time not provided';
    }
    function rel(value) {
        const d = dateOf(value);
        if (!d) return '—';
        const mins = Math.max(0, Math.round((Date.now()-d.getTime())/60000));
        if (mins < 1) return 'just now';
        if (mins < 60) return `${mins}m ago`;
        if (mins < 1440) return `${Math.round(mins/60)}h ago`;
        return `${Math.round(mins/1440)}d ago`;
    }
    const cats={rainfall:'Rainfall',thunderstorm:'Thunderstorm',flooding:'Flooding',heatwave:'Heatwave',fog:'Fog',dust_storm:'Dust storm',strong_winds:'Strong winds',other:'Other'};
    const statuses={VERIFIED_OFFICIAL:'Official source',SOURCE_VERIFIED:'Source verified',CORROBORATED:'Corroborated',NEEDS_REVIEW:'Needs review',LOW_CONFIDENCE:'Low confidence',UNVERIFIED_CITIZEN:'Unverified citizen'};
    const sourceLabels={official:'Official',public_dataset:'Public dataset',weather_api:'Weather API',social:'Social',web:'Public web',citizen:'Citizen'};
    const reviewStatuses=['CORROBORATED','VERIFIED_OFFICIAL','NEEDS_REVIEW','LOW_CONFIDENCE','UNVERIFIED_CITIZEN'];
    const labelCat=v=>cats[v]||String(v||'Other').replace(/_/g,' ');
    const labelStatus=v=>statuses[v]||String(v||'Unknown').replace(/_/g,' ');
    const statusClass=v=>['VERIFIED_OFFICIAL','SOURCE_VERIFIED','CORROBORATED'].includes(v)?'official':(v==='LOW_CONFIDENCE'?'low':'review');

    function adminToken(){ return sessionStorage.getItem('rakshanet_admin_token')||''; }
    function isValidCoord(event){
        const lat=Number(event?.latitude), lon=Number(event?.longitude);
        return Number.isFinite(lat)&&Number.isFinite(lon)&&lat>=-90&&lat<=90&&lon>=-180&&lon<=180&&!(lat===0&&lon===0);
    }
    function locationText(event){
        const bits=[event?.city,event?.district,event?.state].filter(Boolean);
        return bits.length?[...new Set(bits)].join(', '):'Location not available from record';
    }
    function locationBasis(event){
        const basis=event?.metadata?.locationBasis;
        if(basis==='SOURCE_GEOMETRY') return 'Source geometry';
        if(basis==='DERIVED_GEOCODE') return 'Derived from named place';
        if(basis==='DERIVED_DISTRICT_GEOCODE') return 'Derived from district';
        if(basis==='TEXT_MENTION') return 'Mentioned in source text';
        return 'Location basis not provided';
    }
    function screeningText(event){
        if(event?.sourceType==='official') return 'Source verified';
        return Number.isFinite(Number(event?.trustScore))?`Screening ${Math.round(Number(event.trustScore))}/100`:'Screening not scored';
    }
    function hazards(event){
        const values=Array.isArray(event?.metadata?.hazards)?event.metadata.hazards:[];
        return [...new Set(values)].slice(0,4);
    }
    function mentionedPlaces(event){
        const values=Array.isArray(event?.metadata?.mentionedPlaces)?event.metadata.mentionedPlaces:[];
        return [...new Set(values)].filter(Boolean).slice(0,20);
    }
    function connectorClass(status){
        if(status==='CONNECTED'||status==='LOCAL_FALLBACK') return 'live';
        if(['NO_DATA','RATE_LIMITED','ACCESS_BLOCKED'].includes(status)) return 'warn';
        return 'offline';
    }
    function connectorLabel(status){
        return {CONNECTED:'CONNECTED',LOCAL_FALLBACK:'LOCAL FALLBACK',NO_DATA:'NO DATA',UNAVAILABLE:'UNAVAILABLE',RATE_LIMITED:'RATE LIMITED',ACCESS_BLOCKED:'ACCESS BLOCKED'}[status]||status||'UNAVAILABLE';
    }

    async function loadOverview(force=false){
        const btn=$('pulseRefresh');
        if(btn){btn.disabled=true;btn.textContent='⏳ Updating sources…';}
        try{
            const response=await fetch(force?'/api/pulse/overview?refresh=1':'/api/pulse/overview',{cache:'no-store'});
            const data=await response.json();
            if(!response.ok||!data.ok) throw new Error(data.error||`Pulse API HTTP ${response.status}`);
            state.overview=data;
            state.events=Array.isArray(data.events)?data.events:[];
            renderOverview(data);
            await applyFilters();
        }catch(error){
            console.error('Pulse overview failed:',error);
            $('pulseEventFeed').innerHTML=`<div class="pulse-empty">Live ingestion failed: ${esc(error.message)}<br><small>Unavailable sources are not replaced with demo records.</small></div>`;
            $('pulseLastUpdate').textContent=`Live source error · ${error.message}`;
        }finally{
            if(btn){btn.disabled=false;btn.textContent='↻ Refresh live sources';}
        }
    }

    function renderOverview(data){
        const c=data.counts||{};
        $('kpiTotal').textContent=Number(c.ingested||0).toLocaleString('en-IN');
        $('kpiOfficial').textContent=Number(c.officialAlerts||0).toLocaleString('en-IN');
        const officialSources=c.officialAlertSources||{};
        if($('kpiOfficialDetail')) $('kpiOfficialDetail').textContent=`SACHET ${Number(officialSources.sachet||0)} · IMD ${Number(officialSources.imd||0)}`;
        $('kpiSocial').textContent=Number(c.socialSignals||0).toLocaleString('en-IN');
        $('kpiWeb').textContent=Number(c.webSignals||0).toLocaleString('en-IN');
        $('kpiCitizen').textContent=Number(c.citizenReports||0).toLocaleString('en-IN');
        $('kpiReview').textContent=Number(c.needsReview||0).toLocaleString('en-IN');
        const q=data.dataQuality||{};
        $('qualityMapped').textContent=Number(q.mappedRecords||0).toLocaleString('en-IN');
        $('qualityDerived').textContent=Number(q.derivedLocationRecords||0).toLocaleString('en-IN');
        $('qualityMedia').textContent=Number(q.recordsWithMedia||0).toLocaleString('en-IN');
        $('qualityApi').textContent=Number(c.weatherApiObservations||0).toLocaleString('en-IN');
        $('qualityDataset').textContent=Number(c.publicDatasetRecords||0).toLocaleString('en-IN');
        $('heroMapped').textContent=Number(q.mappedRecords||0).toLocaleString('en-IN');
        $('storageMode').textContent=data.storage?.mode||'Not reported';
        const screened = Number(c.recordsScreened ?? c.ingested ?? 0);
        const newRecords = Number(c.newRecords ?? c.inserted ?? 0);
        const duplicateCandidates = Number(c.duplicateCandidates ?? c.duplicatesDetected ?? 0);
        $('pulseLastUpdate').textContent=`Checked ${fmt(data.checkedAt)} · ${Number(data.latencyMs||0)} ms · ${screened} records screened · ${newRecords} new · ${duplicateCandidates} duplicate candidates`;
        renderCategoryChart(data.categoryCounts||{});
        renderStateChart(data.stateCounts||{});
        renderSourceMix(data.sourceCounts||{});
        renderHealth(data.sourceHealth||[]);
        renderObservations(data);
        renderOfficialAlerts(state.events);
        renderMap(state.events);
    }
    function renderCategoryChart(counts){
        const target=$('pulseCategoryChart');
        const entries=Object.entries(counts).sort((a,b)=>Number(b[1])-Number(a[1]));
        if(!entries.length){target.innerHTML='<div class="pulse-muted">No indexed event categories yet.</div>';return;}
        const max=Math.max(1,...entries.map(x=>Number(x[1])));
        target.innerHTML=entries.map(([k,v])=>`<div class="pulse-bar-row"><div class="pulse-bar-label">${esc(labelCat(k))}</div><div class="pulse-bar-track"><div class="pulse-bar-fill" style="width:${Math.max(3,Number(v)/max*100)}%"></div></div><div class="pulse-bar-value">${Number(v)}</div></div>`).join('');
    }
    function renderStateChart(counts){
        const target=$('pulseStateChart');
        const entries=Object.entries(counts).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,10);
        if(!entries.length){target.innerHTML='<div class="pulse-muted">No state metadata is available for current records.</div>';return;}
        const max=Math.max(1,...entries.map(x=>Number(x[1])));
        target.innerHTML=entries.map(([k,v])=>`<div class="pulse-bar-row"><div class="pulse-bar-label">${esc(k)}</div><div class="pulse-bar-track"><div class="pulse-bar-fill" style="width:${Math.max(3,Number(v)/max*100)}%"></div></div><div class="pulse-bar-value">${Number(v)}</div></div>`).join('');
    }
    function renderSourceMix(counts){
        const target=$('pulseSourceMix');
        const entries=Object.entries(counts).sort((a,b)=>Number(b[1])-Number(a[1]));
        target.innerHTML=entries.length?entries.map(([k,v])=>`<div class="pulse-source-row"><div><strong>${esc(sourceLabels[k]||k)}</strong></div><span>${Number(v)} records</span></div>`).join(''):'<div class="pulse-muted">No source records indexed.</div>';
    }
    function renderObservations(data){
        const apiTarget=$('pulseApiObservations');
        const dsTarget=$('pulseDatasetObservations');
        const api=Array.isArray(data.apiObservations)?data.apiObservations:[];
        const ds=Array.isArray(data.datasetObservations)?data.datasetObservations:[];
        if(apiTarget) apiTarget.innerHTML=api.length?api.slice(0,10).map(item=>`<div class="pulse-observation-item"><div><strong>${esc(item.city)}, ${esc(item.state)}</strong><span>${esc(item.time||'Time not provided')}</span></div><div class="pulse-observation-values"><span>${item.temperature??'—'} °C</span><span>${item.humidity??'—'}% RH</span><span>${item.precipitation??'—'} mm</span><span>${item.windSpeed??'—'} km/h</span><span>${esc(item.weatherDescription||'Weather code not provided')}</span></div></div>`).join(''):'<div class="pulse-muted">No public API observations available.</div>';
        if(dsTarget) dsTarget.innerHTML=ds.length?ds.slice(0,10).map(item=>`<div class="pulse-observation-item"><div><strong>${esc(item.title)}</strong><span>${esc(fmt(item.publishedAt))}</span></div><div class="pulse-observation-values"><span>${esc(item.observationType||'Observation')}</span><span>${item.value??'—'} ${esc(item.unit||'')}</span><span>${esc([item.district,item.state].filter(Boolean).join(', ')||'Location not provided')}</span></div></div>`).join(''):'<div class="pulse-muted">No usable public dataset observations are available.</div>';
    }

    function renderHealth(items){
        $('pulseSourceHealth').innerHTML=items.length?items.map(item=>{
            const status=item.status|| (item.available?'CONNECTED':'UNAVAILABLE');
            const tone=['CONNECTED','LOCAL_FALLBACK'].includes(status)?'official':(['NO_DATA','RATE_LIMITED','ACCESS_BLOCKED'].includes(status)?'review':'low');
            return `<div class="pulse-health-item"><div class="pulse-health-top"><div><b class="${connectorClass(status)}"></b><strong>${esc(item.label)}</strong></div><span class="pulse-tag ${tone}">${esc(connectorLabel(status))}</span></div><small>${esc(item.type||'Source')} · ${esc(item.detail||'No detail')} ${item.checkedAt?`· checked ${esc(rel(item.checkedAt))}`:''}</small></div>`;
        }).join(''):'<div class="pulse-muted">No connector status.</div>';
    }
    function mediaMarkup(event){
        const media=Array.isArray(event?.media)?event.media:[];
        const links=media.filter(m=>m&&/^https?:/i.test(String(m.url||'')) || m?.url?.startsWith('/')).slice(0,3);
        if(!links.length) return media.length?`<span class="pulse-tag">Media ${media.length}</span>`:'';
        return links.map((m,i)=>`<a class="pulse-media-link" href="${esc(m.url)}" target="_blank" rel="noreferrer">${m.type==='video'?'Video':'Media'} ${i+1} ↗</a>`).join('');
    }
    function renderOfficialAlerts(events){
        const official=events.filter(e=>e.sourceType==='official').slice(0,10);
        $('pulseOfficialList').innerHTML=official.length?official.map(event=>{
            const hz=hazards(event);
            return `<div class="pulse-alert-item"><div class="pulse-alert-top"><div><div class="pulse-alert-title">${esc(event.title||'Official weather alert')}</div><div class="pulse-meta-row"><span class="pulse-tag official">${esc(labelCat(event.eventCategory))}</span><span class="pulse-tag">${esc(rel(event.publishedAt||event.ingestedAt))}</span>${event.state?`<span class="pulse-tag">${esc(event.state)}</span>`:''}</div></div><span class="pulse-tag official">Source verified</span></div><div class="pulse-review-text">${esc((event.content||'').slice(0,280))}</div><div class="pulse-meta-row"><span class="pulse-tag">${esc(locationText(event))}</span><span class="pulse-tag">${esc(locationBasis(event))}</span>${hz.slice(0,2).map(h=>`<span class="pulse-tag">${esc(labelCat(h))}</span>`).join('')}</div>${mediaMarkup(event)}${event.sourceUrl&&/^https?:/i.test(event.sourceUrl)?`<div style="margin-top:6px"><a href="${esc(event.sourceUrl)}" target="_blank" rel="noreferrer">Open source ↗</a></div>`:''}</div>`;
        }).join(''):'<div class="pulse-empty">No official alert records are available from the connected feeds.</div>';
    }
    function renderEventFeed(events){
        const target=$('pulseEventFeed');
        $('pulseEventCount').textContent=`${events.length} event${events.length===1?'':'s'}`;
        if(!events.length){target.innerHTML='<div class="pulse-empty">No events match the current filters.</div>';return;}
        target.innerHTML=events.slice(0,150).map(event=>{
            const gps=isValidCoord(event);
            const mentioned=mentionedPlaces(event);
            const multi=event.metadata?.locationScope==='MULTI_AREA'||mentioned.length>1;
            const coverage=multi?`Coverage: ${Math.max(mentioned.length,1)} named area${mentioned.length===1?'':'s'}`:'';
            const hz=hazards(event);
            const extra=hz.filter(h=>h!==event.eventCategory);
            return `<article class="pulse-event-item"><div class="pulse-event-top"><div style="min-width:0"><div class="pulse-event-title">${esc(event.title||'Weather event')}</div><div class="pulse-meta-row"><span class="pulse-tag ${statusClass(event.verificationStatus)}">${esc(labelStatus(event.verificationStatus))}</span><span class="pulse-tag">${esc(labelCat(event.eventCategory))}</span><span class="pulse-tag">${esc(screeningText(event))}</span></div></div><div class="pulse-tag">${esc(rel(event.publishedAt||event.ingestedAt))}</div></div><div class="pulse-event-content">${esc((event.content||'').slice(0,420))}</div><div class="pulse-meta-row"><span class="pulse-tag">${esc(sourceLabels[event.sourceType]||event.sourceType||'Source')} · ${esc(event.sourceName||'Unnamed source')}</span><span class="pulse-tag">📍 ${esc(locationText(event))}</span><span class="pulse-tag ${String(event.metadata?.locationBasis||'').startsWith('DERIVED')?'derived':''}">${esc(gps?`${Number(event.latitude).toFixed(4)}, ${Number(event.longitude).toFixed(4)}`:'Coordinates unavailable')}</span>${coverage?`<span class="pulse-tag">${esc(coverage)}</span>`:''}${event.author?`<span class="pulse-tag">By ${esc(event.author)}</span>`:''}${mediaMarkup(event)}</div><div class="pulse-meta-row">${hz.map(h=>`<span class="pulse-tag">Hazard: ${esc(labelCat(h))}</span>`).join('')}</div><details class="pulse-trace"><summary>Traceability details</summary><div class="pulse-review-text"><strong>Published:</strong> ${esc(fmt(event.publishedAt))}<br><strong>Ingested:</strong> ${esc(fmt(event.ingestedAt))}<br><strong>Location basis:</strong> ${esc(locationBasis(event))}<br><strong>Source ID:</strong> ${esc(event.sourceId||'Not provided')}<br><strong>Verification:</strong> ${esc(labelStatus(event.verificationStatus))}${event.sourceUrl?`<br><strong>Source URL:</strong> ${esc(event.sourceUrl)}`:''}${mentioned.length?`<br><strong>Named areas:</strong> ${esc(mentioned.join(', '))}`:''}</div></details></article>`;
        }).join('');
    }
    function adminControls(event){
        const token=adminToken();
        if(!token) return '<a class="pulse-inline-link" href="admin-login.html">🔐 Sign in to verify ↗</a>';
        const selected=String(event.verificationStatus||'NEEDS_REVIEW');
        return `<div class="pulse-admin-review"><select data-review-select="${esc(event.id)}" aria-label="Verification status">${reviewStatuses.map(v=>`<option value="${v}" ${v===selected?'selected':''}>${esc(labelStatus(v))}</option>`).join('')}</select><button class="pulse-review-button" data-review-id="${esc(event.id)}">Save status</button></div>`;
    }
    async function updateVerification(id,select){
        const token=adminToken(); if(!token){window.location.href='admin-login.html';return;}
        const status=select?.value; if(!reviewStatuses.includes(status)) return;
        const button=document.querySelector(`[data-review-id="${CSS.escape(id)}"]`);
        if(button){button.disabled=true;button.textContent='Saving…';}
        try{
            const response=await fetch(`/api/pulse/events/${encodeURIComponent(id)}/verification`,{method:'PATCH',headers:{'Content-Type':'application/json','Authorization':`Bearer ${token}`},body:JSON.stringify({verificationStatus:status})});
            const data=await response.json().catch(()=>({}));
            if(!response.ok||!data.ok) throw new Error(data.error||`Verification update failed (${response.status})`);
            await loadOverview(true);
        }catch(error){
            console.error('Pulse verification update failed:',error);
            if(/token|admin/i.test(String(error.message||''))){sessionStorage.removeItem('rakshanet_admin_token');sessionStorage.removeItem('rakshanet_admin_user');}
            alert(`Could not update this record. ${error.message}`);
        }finally{if(button){button.disabled=false;button.textContent='Save status';}}
    }
    function bindReviewControls(){
        document.querySelectorAll('[data-review-id]').forEach(button=>button.addEventListener('click',()=>updateVerification(button.dataset.reviewId,document.querySelector(`[data-review-select="${CSS.escape(button.dataset.reviewId)}"]`))));
    }
    function renderReview(events){
        const items=events.filter(e=>['social','web'].includes(e.sourceType)).slice(0,30);
        $('pulseReviewFeed').innerHTML=items.length?items.map(event=>`<div class="pulse-review-item"><div class="pulse-review-top"><div><div class="pulse-review-title">${esc(event.title||'Public signal')}</div><div class="pulse-meta-row"><span class="pulse-tag ${statusClass(event.verificationStatus)}">${esc(labelStatus(event.verificationStatus))}</span><span class="pulse-tag">${esc(screeningText(event))}</span><span class="pulse-tag">${esc(labelCat(event.eventCategory))}</span></div></div><span class="pulse-tag">${esc(rel(event.publishedAt||event.ingestedAt))}</span></div><div class="pulse-review-text">${esc((event.content||'').slice(0,300))}</div><div class="pulse-meta-row"><span class="pulse-tag">${esc(event.sourceName||event.sourceType)}</span><span class="pulse-tag">${esc(locationText(event))}</span><span class="pulse-tag">${esc(locationBasis(event))}</span>${mediaMarkup(event)}</div>${event.sourceUrl&&/^https?:/i.test(event.sourceUrl)?`<div style="margin-top:6px"><a href="${esc(event.sourceUrl)}" target="_blank" rel="noreferrer">Open original ↗</a></div>`:''}<div class="pulse-review-actions">${adminControls(event)}</div></div>`).join(''):'<div class="pulse-empty">No public social/web signals are currently indexed.</div>';
        bindReviewControls();
    }
    function renderCitizen(events){
        const items=events.filter(e=>e.sourceType==='citizen').slice(0,30);
        $('pulseCitizenFeed').innerHTML=items.length?items.map(event=>`<div class="pulse-review-item"><div class="pulse-review-top"><div><div class="pulse-review-title">${esc(event.title||'Citizen report')}</div><div class="pulse-meta-row"><span class="pulse-tag review">${esc(labelStatus(event.verificationStatus))}</span><span class="pulse-tag">${esc(screeningText(event))}</span><span class="pulse-tag">${esc(labelCat(event.eventCategory))}</span></div></div><span class="pulse-tag">${esc(rel(event.publishedAt||event.ingestedAt))}</span></div><div class="pulse-review-text">${esc((event.content||'').slice(0,300))}</div><div class="pulse-meta-row"><span class="pulse-tag">${esc(locationText(event))}</span><span class="pulse-tag">${esc(isValidCoord(event)?`${Number(event.latitude).toFixed(4)}, ${Number(event.longitude).toFixed(4)}`:'Coordinates unavailable')}</span>${mediaMarkup(event)}</div>${event.sourceUrl&&/^https?:/i.test(event.sourceUrl)?`<div style="margin-top:6px"><a href="${esc(event.sourceUrl)}" target="_blank" rel="noreferrer">Open report ↗</a></div>`:''}<div class="pulse-review-actions">${adminControls(event)}</div></div>`).join(''):'<div class="pulse-empty">No citizen weather/disaster reports are currently indexed.</div>';
        bindReviewControls();
    }

    function mapFeatures(events){
        return events.filter(isValidCoord).slice(0,500).map(event=>({
            type:'Feature',
            geometry:{type:'Point',coordinates:[Number(event.longitude),Number(event.latitude)]},
            properties:{
                id:String(event.id||''), title:String(event.title||'Weather event'), category:String(event.eventCategory||'other'),
                sourceType:String(event.sourceType||'other'), sourceName:String(event.sourceName||''), status:String(event.verificationStatus||''),
                location:locationText(event), basis:locationBasis(event), content:String(event.content||'').slice(0,500),
                published:String(event.publishedAt||event.ingestedAt||''), sourceUrl:String(event.sourceUrl||'')
            }
        }));
    }
    function popupHtml(p){
        const safeSource=/^https?:/i.test(p.sourceUrl)?`<br><a href="${esc(p.sourceUrl)}" target="_blank" rel="noreferrer">Open source ↗</a>`:'';
        return `<div style="min-width:230px"><strong>${esc(p.title)}</strong><br><span>${esc(labelCat(p.category))} · ${esc(labelStatus(p.status))}</span><br><span>${esc(p.location)}</span><br><em>${esc(p.basis)}</em><br><small>${esc(fmt(p.published))}${safeSource}</small></div>`;
    }
    function addMapLayers(){
        const map=state.map;
        if(!map||!state.mapReady||!map.isStyleLoaded()) return;
        if(!map.getSource('pulse-events')){
            map.addSource('pulse-events',{type:'geojson',data:{type:'FeatureCollection',features:[]},cluster:true,clusterMaxZoom:7,clusterRadius:45});
        }
        if(!map.getLayer('pulse-clusters')) map.addLayer({id:'pulse-clusters',type:'circle',source:'pulse-events',filter:['has','point_count'],paint:{'circle-color':['step',['get','point_count'],'#38d996',10,'#ffc857',30,'#ff6470'],'circle-radius':['step',['get','point_count'],17,10,22,30,28],'circle-stroke-width':2,'circle-stroke-color':'#07111f','circle-opacity':0.86}});
        if(!map.getLayer('pulse-cluster-count')) map.addLayer({id:'pulse-cluster-count',type:'symbol',source:'pulse-events',filter:['has','point_count'],layout:{'text-field':['get','point_count_abbreviated'],'text-size':11},paint:{'text-color':'#07111f'}});
        if(!map.getLayer('pulse-event-points')) map.addLayer({id:'pulse-event-points',type:'circle',source:'pulse-events',filter:['!', ['has','point_count']],paint:{'circle-color':['match',['get','category'],'flooding','#ff6470','thunderstorm','#ffc857','rainfall','#35c8ff','heatwave','#ff9f68','fog','#b8c7d9','dust_storm','#d9b36c','strong_winds','#a5e07c','#8b9bad'],'circle-radius':7,'circle-opacity':0.86,'circle-stroke-width':2,'circle-stroke-color':['match',['get','sourceType'],'official','#38d996','#ffc857']}});
        const source=map.getSource('pulse-events');
        source.setData({type:'FeatureCollection',features:mapFeatures(state.mapEvents)});
        map.off('click','pulse-event-points',handlePointClick); map.on('click','pulse-event-points',handlePointClick);
        map.off('click','pulse-clusters',handleClusterClick); map.on('click','pulse-clusters',handleClusterClick);
        if(!state.mapPointerHandlersBound){
            map.on('mouseenter','pulse-event-points',()=>map.getCanvas().style.cursor='pointer');
            map.on('mouseleave','pulse-event-points',()=>map.getCanvas().style.cursor='');
            map.on('mouseenter','pulse-clusters',()=>map.getCanvas().style.cursor='pointer');
            map.on('mouseleave','pulse-clusters',()=>map.getCanvas().style.cursor='');
            state.mapPointerHandlersBound=true;
        }
        if(state.mapFitPending) fitMapToVisible();
        state.mapFitPending=false;
    }
    function handlePointClick(e){
        const feature=e.features?.[0]; if(!feature||!state.map) return;
        new maplibregl.Popup({offset:12,maxWidth:'360px'}).setLngLat(feature.geometry.coordinates).setHTML(popupHtml(feature.properties||{})).addTo(state.map);
    }
    function handleClusterClick(e){
        const feature=e.features?.[0]; if(!feature||!state.map) return;
        const clusterId=feature.properties.cluster_id;
        const src=state.map.getSource('pulse-events');
        src.getClusterExpansionZoom(clusterId).then(zoom=>state.map.easeTo({center:feature.geometry.coordinates,zoom})).catch(()=>{});
    }
    function fitMapToVisible(){
        if(!state.map||!state.mapReady||!state.map.isStyleLoaded()) return;
        const features=mapFeatures(state.mapEvents);
        if(!features.length){state.map.easeTo({center:INDIA_CENTER,zoom:4.7});$('pulseMapNote').textContent='No map-ready coordinates are available for the current filters. No location is fabricated.';return;}
        const bounds=new maplibregl.LngLatBounds();
        features.forEach(f=>bounds.extend(f.geometry.coordinates));
        state.map.fitBounds(bounds,{padding:40,maxZoom:8,duration:450});
        const derived=features.filter(f=>String(f.properties.basis||'').startsWith('Derived')).length;
        $('pulseMapNote').textContent=`${features.length} map-ready record${features.length===1?'':'s'} shown · ${derived} use derived coordinates. Multi-area alerts are mapped only when source geometry is available.`;
    }
    async function loadOpenFreeMapStyle(){
        const response=await fetch(OPENFREEMAP_STYLE,{cache:'force-cache'});
        if(!response.ok) throw new Error(`OpenFreeMap style HTTP ${response.status}`);
        const style=await response.json();
        // OpenFreeMap's current style can advertise an SDF glyph template that returns
        // 404 for the combined local font stack. MapLibre permits glyphs to be omitted,
        // in which case browser-local fonts are used. Keep all source/sprite references
        // resolvable against the actual style URL before committing the style object.
        delete style.glyphs;
        const base=new URL(OPENFREEMAP_STYLE);
        if(style.sprite) style.sprite=new URL(style.sprite,base).href;
        for(const source of Object.values(style.sources||{})){
            if(source && source.url) source.url=new URL(source.url,base).href;
            if(source && Array.isArray(source.tiles)) source.tiles=source.tiles.map(tile=>new URL(tile,base).href);
        }
        style.metadata={...(style.metadata||{}),'rakshanet:glyphs':'local-browser-fonts'};
        return style;
    }

    async function initMap(){
        const el=$('pulseMap'); if(!el) return;
        try{
            let mapStyle;
            try{
                mapStyle=await loadOpenFreeMapStyle();
            }catch(styleError){
                console.warn('OpenFreeMap style unavailable; using OpenStreetMap raster fallback.',styleError);
                state.mapStyleFallback=true;
                mapStyle=OSM_RASTER_STYLE;
                $('pulseMapProvider').textContent='MapLibre · OpenStreetMap fallback';
            }
            state.map=new maplibregl.Map({
                container:el,
                style:mapStyle,
                center:INDIA_CENTER,
                zoom:4.7,
                minZoom:3,
                maxZoom:12,
                attributionControl:false,
                localIdeographFontFamily:'Arial, sans-serif'
            });
            state.map.addControl(new maplibregl.NavigationControl({showCompass:true}), 'top-right');
            state.map.addControl(new maplibregl.ScaleControl({maxWidth:120,unit:'metric'}), 'bottom-left');
            state.map.addControl(new maplibregl.AttributionControl({compact:true}), 'bottom-right');
            state.map.on('load',()=>{state.mapReady=true;addMapLayers();fitMapToVisible();});
            state.map.on('style.load',()=>{if(state.mapReady)addMapLayers();});
            state.map.on('error',event=>{
                const status=Number(event?.error?.status||0); const msg=String(event?.error?.message||'').toLowerCase();
                if(!state.mapStyleFallback && (status===401||status===403||status===404||msg.includes('openfreemap')||msg.includes('style'))){
                    state.mapStyleFallback=true;
                    $('pulseMapProvider').textContent='MapLibre · OpenStreetMap fallback';
                    state.map.setStyle(OSM_RASTER_STYLE);
                }
            });
            if(!state.mapStyleFallback) $('pulseMapProvider').textContent='MapLibre · OpenFreeMap (OSM data · local fonts)';
        }catch(error){
            console.error('Map initialization failed:',error);
            $('pulseMap').innerHTML='<div class="pulse-empty">Interactive map could not be initialized. Event data remains available above and coordinates are not fabricated.</div>';
        }
    }
    function renderMap(events){
        state.mapEvents=events;
        state.mapFitPending=true;
        addMapLayers();
        if(state.mapReady) fitMapToVisible();
    }

    function buildFilterParams(){
        const p=new URLSearchParams();
        const range=$('pulseTimeFilter').value;
        if(range!=='all'&&range!=='custom'){
            const hours=range==='1h'?1:range==='6h'?6:range==='24h'?24:168;
            p.set('from',new Date(Date.now()-hours*3600000).toISOString());
        }
        if(range==='custom'){
            const from=$('pulseFromDate').value, to=$('pulseToDate').value;
            if(from)p.set('from',new Date(from).toISOString());
            if(to)p.set('to',new Date(to).toISOString());
        }
        const ev=$('pulseEventFilter').value, src=$('pulseSourceFilter').value, st=$('pulseStateFilter').value.trim(), ver=$('pulseVerificationFilter').value, q=$('pulseSearch').value.trim();
        if(ev!=='all')p.set('event',ev); if(src!=='all')p.set('source',src); if(st)p.set('state',st); if(ver!=='ALL')p.set('verification',ver); if(q)p.set('q',q);
        return p;
    }
    async function applyFilters(){
        if(state.filterBusy)return;
        state.filterBusy=true;
        try{
            const p=buildFilterParams();
            const response=await fetch(`/api/pulse/events?${p.toString()}`,{cache:'no-store'});
            const data=await response.json();
            if(!response.ok||!data.ok)throw new Error(data.error||`Filter HTTP ${response.status}`);
            const events=Array.isArray(data.events)?data.events:[];
            renderEventFeed(events);renderReview(events);renderCitizen(events);renderOfficialAlerts(events);renderMap(events);
            $('pulseFilterStatus').textContent=`${events.length} matching events`;
        }catch(error){console.error('Pulse filter failed:',error);}
        finally{state.filterBusy=false;}
    }
    function setupControls(){
        $('pulseTimeFilter')?.addEventListener('change',()=>{
            const custom=$('pulseTimeFilter').value==='custom';$('pulseFromDate').disabled=!custom;$('pulseToDate').disabled=!custom;applyFilters();
        });
        ['pulseEventFilter','pulseSourceFilter','pulseVerificationFilter'].forEach(id=>$(id)?.addEventListener('change',applyFilters));
        ['pulseFromDate','pulseToDate'].forEach(id=>$(id)?.addEventListener('change',()=>{if($('pulseTimeFilter').value==='custom')applyFilters();}));
        $('pulseApplyFilters')?.addEventListener('click',applyFilters);
        $('pulseSearch')?.addEventListener('keydown',e=>{if(e.key==='Enter')applyFilters();});
        $('pulseStateFilter')?.addEventListener('keydown',e=>{if(e.key==='Enter')applyFilters();});
        $('pulseRefresh')?.addEventListener('click',()=>loadOverview(true));
        $('pulseFitMap')?.addEventListener('click',fitMapToVisible);
        $('pulseTopButton')?.addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));
        window.addEventListener('scroll',()=>{$('pulseTopButton')?.classList.toggle('visible',window.scrollY>500);},{passive:true});
    }

    setupControls();
    initMap().finally(()=>loadOverview(false));
    setInterval(()=>loadOverview(false),5*60*1000);
})();
