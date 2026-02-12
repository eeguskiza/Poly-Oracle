#!/bin/bash

echo "========================================"
echo "Poly-Oracle News System Test Suite"
echo "========================================"
echo ""

# Test 1: Búsqueda simple de noticias
echo "Test 1: Búsqueda de noticias sobre 'Trump'"
echo "----------------------------------------"
./venv/bin/python cli.py news "Trump" --limit 3 2>&1 | grep -v "INFO\|DEBUG\|WARNING"
echo ""

# Test 2: Búsqueda de noticias sobre tecnología
echo "Test 2: Búsqueda de noticias sobre 'SpaceX'"
echo "----------------------------------------"
./venv/bin/python cli.py news "SpaceX" --limit 3 2>&1 | grep -v "INFO\|DEBUG\|WARNING"
echo ""

# Test 3: Búsqueda de noticias sobre economía
echo "Test 3: Búsqueda de noticias sobre 'economy inflation'"
echo "----------------------------------------"
./venv/bin/python cli.py news "economy inflation" --limit 3 2>&1 | grep -v "INFO\|DEBUG\|WARNING"
echo ""

# Test 4: Ver markets disponibles
echo "Test 4: Listar markets disponibles"
echo "----------------------------------------"
./venv/bin/python cli.py markets --limit 3 2>&1 | grep -v "INFO\|DEBUG\|WARNING"
echo ""

# Test 5: Noticias para un market específico
echo "Test 5: Noticias relevantes para market 517310"
echo "----------------------------------------"
./venv/bin/python cli.py market-news 517310 2>&1 | grep -v "INFO\|DEBUG\|WARNING" | head -15
echo ""

# Test 6: Detalles de un market
echo "Test 6: Detalles del market 517310"
echo "----------------------------------------"
./venv/bin/python cli.py market 517310 2>&1 | grep -v "INFO\|DEBUG\|WARNING" | head -10
echo ""

echo "========================================"
echo "Tests completados!"
echo "========================================"
