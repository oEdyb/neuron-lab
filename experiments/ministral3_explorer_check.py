"""Headless smoke check of the exported Ministral map and its local API."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out'
MODEL='mistralai/Ministral-3-3B-Instruct-2512'
URL='http://127.0.0.1:8794'


def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        try:
            page=browser.new_page(viewport=dict(width=1440,height=1000))
            errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(URL)
            page.wait_for_function('window.neuronNext && window.neuronNext.stats().layers === 26',timeout=60000)
            state=page.request.get(URL+'/api/state').json()
            status=page.request.get(URL+'/api/maps').json()
            manifest=page.request.get(URL+'/data/manifest.json').json()
            assert state['model'] is None
            assert state['configured_model']==MODEL
            assert status['available'] is True
            assert manifest['model']==MODEL
            assert len(manifest['layers'])==26
            assert sum(l['count'] for l in manifest['layers'])==239616
            assert all(l['count']==9216 for l in manifest['layers'])
            neuron=page.request.get(URL+'/api/neuron?layer=16&n=2922').json()
            assert neuron['sentence_words_available']
            source=page.request.get(URL+'/data/L16.json').json()
            word=next(w.strip() for w in source['rows'][2922]['up'] if len(w.strip())>=3 and w.strip().isalpha())
            page.get_by_role('tab',name='Exact',exact=True).click()
            query=page.get_by_placeholder('Word, meaning, or L17 N5244',exact=True)
            query.fill(word)
            page.locator('.result-row').first.wait_for(timeout=30000)
            exact_count=page.locator('.result-row').count()
            query.fill('L16 N2922')
            page.locator('.result-row .result').filter(has_text='L16 N2922').click()
            page.get_by_role('heading',name='Layer 16 · Neuron 2922',exact=True).wait_for()
            page.screenshot(path=str(OUT/'ministral3-explorer.png'),full_page=True)
            assert not errors,errors
            report=dict(model=MODEL,layers=26,neurons=239616,exact_word=word,exact_results=exact_count,
                        selected=dict(layer=16,n=2922),sentence_words=True,page_errors=errors,
                        stats=page.evaluate('window.neuronNext.stats()'),
                        verified=['26-layer Tower geometry loaded','Exact search','neuron inspection',
                                  'per-sentence words API','model-unloaded map serving'],
                        not_checked=['semantic search','Web connections','local GPU generation'])
            (OUT/'ministral3-explorer-check.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report))
        finally:
            browser.close()


if __name__=='__main__':
    main()
